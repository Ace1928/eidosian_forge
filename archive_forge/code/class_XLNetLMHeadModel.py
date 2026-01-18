import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_xlnet import XLNetConfig
@add_start_docstrings('\n    XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).\n    ', XLNET_START_DOCSTRING)
class XLNetLMHeadModel(XLNetPreTrainedModel):
    _tied_weights_keys = ['lm_loss.weight']

    def __init__(self, config):
        super().__init__(config)
        self.attn_type = config.attn_type
        self.same_length = config.same_length
        self.transformer = XLNetModel(config)
        self.lm_loss = nn.Linear(config.d_model, config.vocab_size, bias=True)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_loss

    def set_output_embeddings(self, new_embeddings):
        self.lm_loss = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_mems=None, **kwargs):
        effective_batch_size = input_ids.shape[0]
        dummy_token = torch.zeros((effective_batch_size, 1), dtype=torch.long, device=input_ids.device)
        offset = 2
        if past_key_values:
            input_ids = torch.cat([input_ids[:, -offset:], dummy_token], dim=1)
        else:
            input_ids = torch.cat([input_ids, dummy_token], dim=1)
        sequence_length = input_ids.shape[1]
        perm_mask = torch.zeros((effective_batch_size, sequence_length, sequence_length), dtype=torch.float, device=input_ids.device)
        perm_mask[:, :, -1] = 1.0
        target_mapping = torch.zeros((effective_batch_size, 1, sequence_length), dtype=torch.float, device=input_ids.device)
        target_mapping[:, 0, -1] = 1.0
        inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping, 'use_mems': use_mems}
        if past_key_values:
            inputs['mems'] = tuple((layer_past[:-offset, :, :] for layer_past in past_key_values))
        return inputs

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=XLNetLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, mems: Optional[torch.Tensor]=None, perm_mask: Optional[torch.Tensor]=None, target_mapping: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, input_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, XLNetLMHeadModelOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, num_predict)`, *optional*):
            Labels for masked language modeling. `num_predict` corresponds to `target_mapping.shape[1]`. If
            `target_mapping` is `None`, then `num_predict` corresponds to `sequence_length`.

            The labels should correspond to the masked input words that should be predicted and depends on
            `target_mapping`. Note in order to perform standard auto-regressive language modeling a *<mask>* token has
            to be added to the `input_ids` (see the `prepare_inputs_for_generation` function and examples below)

            Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored, the loss
            is only computed for labels in `[0, ..., config.vocab_size]`

        Return:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, XLNetLMHeadModel
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")
        >>> model = XLNetLMHeadModel.from_pretrained("xlnet/xlnet-large-cased")

        >>> # We show how to setup inputs to predict a next token using a bi-directional context.
        >>> input_ids = torch.tensor(
        ...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)
        ... ).unsqueeze(
        ...     0
        ... )  # We will predict the masked token
        >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
        >>> target_mapping = torch.zeros(
        ...     (1, 1, input_ids.shape[1]), dtype=torch.float
        ... )  # Shape [1, 1, seq_length] => let's predict one token
        >>> target_mapping[
        ...     0, 0, -1
        ... ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

        >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
        >>> next_token_logits = outputs[
        ...     0
        ... ]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

        >>> # The same way can the XLNetLMHeadModel be used to be trained by standard auto-regressive language modeling.
        >>> input_ids = torch.tensor(
        ...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)
        ... ).unsqueeze(
        ...     0
        ... )  # We will predict the masked token
        >>> labels = torch.tensor(tokenizer.encode("cute", add_special_tokens=False)).unsqueeze(0)
        >>> assert labels.shape[0] == 1, "only one word will be predicted"
        >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        >>> perm_mask[
        ...     :, :, -1
        ... ] = 1.0  # Previous tokens don't see last token as is done in standard auto-regressive lm training
        >>> target_mapping = torch.zeros(
        ...     (1, 1, input_ids.shape[1]), dtype=torch.float
        ... )  # Shape [1, 1, seq_length] => let's predict one token
        >>> target_mapping[
        ...     0, 0, -1
        ... ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

        >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=labels)
        >>> loss = outputs.loss
        >>> next_token_logits = (
        ...     outputs.logits
        ... )  # Logits have shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        logits = self.lm_loss(transformer_outputs[0])
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return XLNetLMHeadModelOutput(loss=loss, logits=logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    @staticmethod
    def _reorder_cache(mems: List[torch.Tensor], beam_idx: torch.Tensor) -> List[torch.Tensor]:
        """
        This function is used to re-order the `mems` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `mems` with the correct beam_idx at every
        generation step.
        """
        return [layer_past.index_select(1, beam_idx.to(layer_past.device)) for layer_past in mems]