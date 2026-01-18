import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation import GenerationConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D
from ...utils import (
from .configuration_clvp import (
@add_start_docstrings('The CLVP decoder model with a language modelling head on top.', CLVP_START_DOCSTRING)
class ClvpForCausalLM(ClvpPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = ClvpModel(self.config)
        self.final_norm = nn.LayerNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=True)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.input_embeds_layer

    def set_input_embeddings(self, new_embeddings):
        self.model.decoder.input_embeds_layer = new_embeddings

    def _prepare_model_inputs(self, inputs: Optional[torch.Tensor]=None, bos_token_id: Optional[int]=None, model_kwargs: Optional[Dict[str, torch.Tensor]]=None) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        input_name = self.main_input_name
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(f'`inputs`: {inputs}` were passed alongside {input_name} which is not allowed.Make sure to either pass {inputs} or {input_name}=...')
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg
        if input_name == 'input_ids' and 'inputs_embeds' in model_kwargs:
            model_kwargs['input_ids'] = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs=model_kwargs)
            inputs, input_name = (model_kwargs['inputs_embeds'], 'inputs_embeds')
        conditioning_embeds = model_kwargs.get('conditioning_embeds', None)
        if conditioning_embeds is not None:
            mel_start_token_embedding = self.model.decoder.input_embeds_layer(torch.full((conditioning_embeds.shape[0], 1), fill_value=self.config.bos_token_id, device=conditioning_embeds.device))
            mel_start_token_embedding += self.model.decoder.position_embeds_layer(torch.full((conditioning_embeds.shape[0], 1), fill_value=0, device=conditioning_embeds.device))
            conditioning_embeds = torch.concat([conditioning_embeds, mel_start_token_embedding], dim=1)
            if hasattr(model_kwargs, 'attention_mask'):
                position_ids = model_kwargs['attention_mask'].long().cumsum(-1) - 1
            else:
                position_ids = torch.range(0, conditioning_embeds.shape[1] - 1, dtype=torch.long, device=conditioning_embeds.device)
            position_ids = position_ids.unsqueeze(0).repeat(conditioning_embeds.shape[0], 1)
            model_kwargs['inputs_embeds'] = conditioning_embeds - self.model.decoder.position_embeds_layer(position_ids)
            model_kwargs['input_ids'] = torch.ones((model_kwargs['inputs_embeds'].shape[0], 1), dtype=torch.long, device=self.device) * self.config.bos_token_id
            return (model_kwargs['inputs_embeds'], 'inputs_embeds', model_kwargs)
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return (inputs, input_name, model_kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, conditioning_embeds=None, **kwargs):
        input_ids_length = input_ids.shape[-1]
        token_type_ids = kwargs.get('token_type_ids', None)
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1]:]
        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        if conditioning_embeds is not None and past_key_values is not None:
            position_ids = torch.tensor([input_ids_length], dtype=torch.long, device=input_ids.device)
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs.update({'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'position_ids': position_ids, 'token_type_ids': token_type_ids})
        return model_inputs

    @add_start_docstrings_to_model_forward(CLVP_DECODER_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        lm_logits = self.final_norm(hidden_states)
        lm_logits = self.lm_head(lm_logits)
        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=lm_logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

    @staticmethod
    def _reorder_cache(past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple((tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)) for layer_past in past_key_values))