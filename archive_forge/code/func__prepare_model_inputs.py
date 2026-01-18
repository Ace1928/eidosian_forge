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