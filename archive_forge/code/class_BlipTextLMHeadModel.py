import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, device, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import (
from ...utils import logging
from .configuration_blip import BlipTextConfig
class BlipTextLMHeadModel(BlipTextPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BlipTextModel(config, add_pooling_layer=False)
        self.cls = BlipTextOnlyMLMHead(config)
        self.label_smoothing = config.label_smoothing

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.Tensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, return_logits: Optional[bool]=False, is_decoder: Optional[bool]=True, reduction: Optional[str]='mean') -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        """
        encoder_hidden_states (`torch.FloatTensor`, *optional*): Sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is
            configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        outputs = self.bert(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, is_decoder=is_decoder)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()
        lm_loss = None
        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous().to(shifted_prediction_scores.device)
            loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=self.label_smoothing)
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            if reduction == 'none':
                lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (lm_loss,) + output if lm_loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=lm_loss, logits=prediction_scores, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'past_key_values': past_key_values, 'encoder_hidden_states': model_kwargs.get('encoder_hidden_states', None), 'encoder_attention_mask': model_kwargs.get('encoder_attention_mask', None), 'is_decoder': True}

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)),)
        return reordered_past