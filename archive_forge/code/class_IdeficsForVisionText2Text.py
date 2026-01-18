from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_idefics import IdeficsConfig
from .perceiver import IdeficsPerceiverResampler
from .vision import IdeficsVisionTransformer
class IdeficsForVisionText2Text(IdeficsPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']
    _tied_weights_keys = ['model.embed_tokens.weight', 'lm_head.weight']

    def __init__(self, config, vision_model=None):
        super().__init__(config)
        self.model = IdeficsModel(config)
        self.lm_head = IdeficsDecoupledLinear(in_features=config.hidden_size, out_features=config.vocab_size, out_additional_features=config.additional_vocab_size, bias=False, partially_freeze=config.freeze_lm_head)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def tie_weights(self):
        """
        Overwrite `transformers.modeling_utils.PreTrainedModel.tie_weights` to handle the case of
        IdeficsDecoupledLinear and IdeficsDecoupledEmbedding.
        """
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()
        if getattr(self.config, 'tie_word_embeddings', True):
            output_embeddings.weight = input_embeddings.weight
            if input_embeddings.num_additional_embeddings > 0:
                assert output_embeddings.out_additional_features == input_embeddings.num_additional_embeddings
                output_embeddings.additional_fc.weight = input_embeddings.additional_embedding.weight
        if hasattr(output_embeddings, 'out_features') and hasattr(input_embeddings, 'num_embeddings'):
            output_embeddings.out_features = input_embeddings.num_embeddings
            if hasattr(output_embeddings, 'out_additional_features') and hasattr(input_embeddings, 'num_additional_embeddings'):
                output_embeddings.out_additional_features = input_embeddings.num_additional_embeddings

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=IdeficsCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, image_encoder_embeddings: Optional[torch.FloatTensor]=None, perceiver_embeddings: Optional[torch.FloatTensor]=None, image_attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, interpolate_pos_encoding: Optional[bool]=False, return_dict: Optional[bool]=None) -> Union[Tuple, IdeficsCausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, IdeficsForVisionText2Text

        >>> model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/idefics-9b")

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\\nI'm not consciours, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, pixel_values=pixel_values, image_encoder_embeddings=image_encoder_embeddings, perceiver_embeddings=perceiver_embeddings, image_attention_mask=image_attention_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, interpolate_pos_encoding=interpolate_pos_encoding, return_dict=return_dict)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return IdeficsCausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, image_hidden_states=outputs.image_hidden_states)

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        image_hidden_states = kwargs.pop('image_hidden_states', None)
        if image_hidden_states is not None:
            if self.config.use_resampler:
                kwargs['perceiver_embeddings'] = image_hidden_states
            else:
                kwargs['image_encoder_embeddings'] = image_hidden_states
            kwargs['pixel_values'] = None
        inputs = prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        unwanted_kwargs = ['token_type_ids']
        for kwarg in unwanted_kwargs:
            inputs.pop(kwarg, None)
        return inputs

    @staticmethod
    def _expand_inputs_for_generation(*args, **model_kwargs):
        return expand_inputs_for_generation(*args, **model_kwargs)

    @staticmethod
    def _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder):
        return update_model_kwargs_for_generation(outputs, model_kwargs)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past)),)
        return reordered_past