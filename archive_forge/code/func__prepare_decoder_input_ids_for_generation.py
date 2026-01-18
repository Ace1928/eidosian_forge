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
def _prepare_decoder_input_ids_for_generation(self, batch_size: int, model_input_name: str, model_kwargs: Dict[str, torch.Tensor], decoder_start_token_id: int=None, bos_token_id: int=None, device: torch.device=None) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
    """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
    if model_kwargs is not None and 'decoder_input_ids' in model_kwargs:
        decoder_input_ids = model_kwargs.pop('decoder_input_ids')
    elif 'input_ids' in model_kwargs and model_input_name != 'input_ids':
        decoder_input_ids = model_kwargs.pop('input_ids')
    else:
        decoder_input_ids = None
    decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
    if device is None:
        device = self.device
    decoder_input_ids_start = torch.ones((batch_size * self.decoder.num_codebooks, 1), dtype=torch.long, device=device) * decoder_start_token_id
    if decoder_input_ids is None:
        decoder_input_ids = decoder_input_ids_start
    elif (decoder_input_ids[..., 0] != decoder_start_token_id).all().item():
        decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
        if 'decoder_attention_mask' in model_kwargs:
            decoder_attention_mask = model_kwargs['decoder_attention_mask']
            decoder_attention_mask = torch.cat((torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask), dim=-1)
            model_kwargs['decoder_attention_mask'] = decoder_attention_mask
    return (decoder_input_ids, model_kwargs)