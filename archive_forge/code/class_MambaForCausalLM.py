import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available
from .configuration_mamba import MambaConfig
from ..deprecated._archive_maps import MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
@add_start_docstrings('\n    The MAMBA Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', MAMBA_START_DOCSTRING)
class MambaForCausalLM(MambaPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config):
        super().__init__(config)
        self.backbone = MambaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        model_kwargs['cache_params'] = outputs.get('cache_params', None)
        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, cache_params: Optional[MambaCache]=None, inputs_embeds=None, attention_mask=None, **kwargs):
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if inputs_embeds is not None and cache_params is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs['cache_params'] = cache_params
        return model_inputs

    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MambaCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, cache_params: Optional[MambaCache]=None, labels: Optional[torch.LongTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, use_cache: Optional[bool]=None, **kwargs) -> Union[Tuple, MambaCausalLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        mamba_outputs = self.backbone(input_ids, cache_params=cache_params, inputs_embeds=inputs_embeds, output_hidden_states=output_hidden_states, return_dict=return_dict, use_cache=use_cache)
        hidden_states = mamba_outputs[0]
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return (loss,) + output if loss is not None else output
        return MambaCausalLMOutput(loss=loss, logits=logits, cache_params=mamba_outputs.cache_params, hidden_states=mamba_outputs.hidden_states)