import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from .configuration_musicgen import MusicgenConfig, MusicgenDecoderConfig
def _maybe_initialize_input_ids_for_generation(self, inputs: Optional[torch.Tensor]=None, bos_token_id: Optional[int]=None, model_kwargs: Optional[Dict[str, torch.Tensor]]=None) -> torch.LongTensor:
    """Initializes input ids for generation, if necessary."""
    if inputs is not None:
        return inputs
    encoder_outputs = model_kwargs.get('encoder_outputs')
    if encoder_outputs is not None:
        shape = encoder_outputs[0].size()[:-1]
        return torch.ones(shape, dtype=torch.long, device=self.device) * -100
    if bos_token_id is None:
        raise ValueError('`bos_token_id` has to be defined when no `input_ids` are provided.')
    batch_size = 1
    for value in model_kwargs.values():
        if isinstance(value, torch.Tensor):
            batch_size = value.shape[0]
            break
    return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id