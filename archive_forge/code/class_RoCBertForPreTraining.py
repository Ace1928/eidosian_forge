import math
import os
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_roc_bert import RoCBertConfig
@add_start_docstrings('\n    RoCBert Model with contrastive loss and masked_lm_loss during the pretraining.\n    ', ROC_BERT_START_DOCSTRING)
class RoCBertForPreTraining(RoCBertPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder.weight', 'cls.predictions.decoder.bias']

    def __init__(self, config):
        super().__init__(config)
        self.roc_bert = RoCBertModel(config)
        self.cls = RoCBertOnlyMLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, input_shape_ids: Optional[torch.Tensor]=None, input_pronunciation_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, attack_input_ids: Optional[torch.Tensor]=None, attack_input_shape_ids: Optional[torch.Tensor]=None, attack_input_pronunciation_ids: Optional[torch.Tensor]=None, attack_attention_mask: Optional[torch.Tensor]=None, attack_token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels_input_ids: Optional[torch.Tensor]=None, labels_input_shape_ids: Optional[torch.Tensor]=None, labels_input_pronunciation_ids: Optional[torch.Tensor]=None, labels_attention_mask: Optional[torch.Tensor]=None, labels_token_type_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """
            attack_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                attack sample ids for computing the contrastive loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            attack_input_shape_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                attack sample shape ids for computing the contrastive loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            attack_input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                attack sample pronunciation ids for computing the contrastive loss. Indices should be in `[-100, 0,
                ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            labels_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                target ids for computing the contrastive loss and masked_lm_loss . Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            labels_input_shape_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                target shape ids for computing the contrastive loss and masked_lm_loss . Indices should be in `[-100,
                0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            labels_input_pronunciation_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                target pronunciation ids for computing the contrastive loss and masked_lm_loss . Indices should be in
                `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                 ignored (masked), the loss is only computed for the tokens with labels in `[0, ...,
                 config.vocab_size]`

            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RoCBertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("weiweishi/roc-bert-base-zh")
        >>> model = RoCBertForPreTraining.from_pretrained("weiweishi/roc-bert-base-zh")

        >>> inputs = tokenizer("你好，很高兴认识你", return_tensors="pt")
        >>> attack_inputs = {}
        >>> for key in list(inputs.keys()):
        ...     attack_inputs[f"attack_{key}"] = inputs[key]
        >>> label_inputs = {}
        >>> for key in list(inputs.keys()):
        ...     label_inputs[f"labels_{key}"] = inputs[key]

        >>> inputs.update(label_inputs)
        >>> inputs.update(attack_inputs)
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> logits.shape
        torch.Size([1, 11, 21128])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roc_bert(input_ids, input_shape_ids=input_shape_ids, input_pronunciation_ids=input_pronunciation_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.cls(sequence_output)
        loss = None
        if labels_input_ids is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels_input_ids.view(-1))
            if attack_input_ids is not None:
                batch_size, _ = labels_input_ids.shape
                device = labels_input_ids.device
                target_inputs = torch.clone(labels_input_ids)
                target_inputs[target_inputs == -100] = self.config.pad_token_id
                labels_output = self.roc_bert(target_inputs, input_shape_ids=labels_input_shape_ids, input_pronunciation_ids=labels_input_pronunciation_ids, attention_mask=labels_attention_mask, token_type_ids=labels_token_type_ids, return_dict=return_dict)
                attack_output = self.roc_bert(attack_input_ids, input_shape_ids=attack_input_shape_ids, input_pronunciation_ids=attack_input_pronunciation_ids, attention_mask=attack_attention_mask, token_type_ids=attack_token_type_ids, return_dict=return_dict)
                labels_pooled_output = labels_output[1]
                attack_pooled_output = attack_output[1]
                pooled_output_norm = torch.nn.functional.normalize(pooled_output, dim=-1)
                labels_pooled_output_norm = torch.nn.functional.normalize(labels_pooled_output, dim=-1)
                attack_pooled_output_norm = torch.nn.functional.normalize(attack_pooled_output, dim=-1)
                sim_matrix = torch.matmul(pooled_output_norm, attack_pooled_output_norm.T)
                sim_matrix_target = torch.matmul(labels_pooled_output_norm, attack_pooled_output_norm.T)
                batch_labels = torch.tensor(list(range(batch_size)), device=device)
                contrastive_loss = (loss_fct(100 * sim_matrix.view(batch_size, -1), batch_labels.view(-1)) + loss_fct(100 * sim_matrix_target.view(batch_size, -1), batch_labels.view(-1))) / 2
                loss = contrastive_loss + masked_lm_loss
            else:
                loss = masked_lm_loss
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return MaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)