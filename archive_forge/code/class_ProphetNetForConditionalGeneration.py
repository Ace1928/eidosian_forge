import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_prophetnet import ProphetNetConfig
@add_start_docstrings('The ProphetNet Model with a language modeling head. Can be used for sequence generation tasks.', PROPHETNET_START_DOCSTRING)
class ProphetNetForConditionalGeneration(ProphetNetPreTrainedModel):
    _tied_weights_keys = ['encoder.word_embeddings.weight', 'decoder.word_embeddings.weight', 'lm_head.weight']

    def __init__(self, config: ProphetNetConfig):
        super().__init__(config)
        self.prophetnet = ProphetNetModel(config)
        self.padding_idx = config.pad_token_id
        self.disable_ngram_loss = config.disable_ngram_loss
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.prophetnet.word_embeddings, self.lm_head)

    def get_input_embeddings(self):
        return self.prophetnet.word_embeddings

    @add_start_docstrings_to_model_forward(PROPHETNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ProphetNetSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.Tensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, inputs_embeds: Optional[torch.Tensor]=None, decoder_inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, ProphetNetSeq2SeqLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, ProphetNetForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
        >>> model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        >>> logits_next_token = outputs.logits  # logits to predict next token as usual
        >>> logits_ngram_next_tokens = outputs.logits_ngram  # logits to predict 2nd, 3rd, ... next tokens
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None and decoder_input_ids is None and (decoder_inputs_embeds is None):
            decoder_input_ids = self._shift_right(labels)
        outputs = self.prophetnet(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, past_key_values=past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        batch_size, sequence_length = decoder_input_ids.shape if decoder_input_ids is not None else decoder_inputs_embeds.shape[:2]
        predicting_streams = outputs[1].view(batch_size, self.config.ngram, sequence_length, -1)
        predict_logits = self.lm_head(predicting_streams)
        logits = predict_logits[:, 0]
        logits_ngram = predict_logits[:, 1:] if self.config.ngram > 1 else None
        if not logits.is_contiguous():
            logits = logits.contiguous()
        loss = None
        if labels is not None:
            loss = self._compute_loss(predict_logits, labels)
        if not return_dict:
            all_logits = tuple((v for v in [logits, logits_ngram] if v is not None))
            return (loss,) + all_logits + outputs[2:] if loss is not None else all_logits + outputs[2:]
        else:
            return ProphetNetSeq2SeqLMOutput(loss=loss, logits=logits, logits_ngram=logits_ngram, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_ngram_hidden_states=outputs.decoder_ngram_hidden_states, decoder_attentions=outputs.decoder_attentions, decoder_ngram_attentions=outputs.decoder_ngram_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    def _compute_loss(self, logits, labels, ignore_index=-100):
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(ignore_index)
        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            expend_targets[i, :, :] = labels
        logits = logits.transpose(0, 1).contiguous()
        lprobs = nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=-1, dtype=torch.float32)
        loss = nn.functional.nll_loss(lprobs, expend_targets.view(-1), reduction='mean')
        if self.config.eps > 0.0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()
            eps_i = self.config.eps / lprobs.size(-1)
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss
        return loss

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        assert encoder_outputs is not None, '`encoder_outputs` have to be passed for generation.'
        if past_key_values:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past

    def get_encoder(self):
        return self.prophetnet.encoder

    def get_decoder(self):
        return self.prophetnet.decoder