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
def _prepare_audio_encoder_kwargs_for_generation(self, input_values, model_kwargs, model_input_name: Optional[str]=None):
    encoder = self.get_audio_encoder()
    if hasattr(encoder, '_hf_hook'):
        encoder._hf_hook.io_same_device = True
    irrelevant_prefix = ['decoder_', 'cross_attn', 'use_cache']
    encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not any((argument.startswith(p) for p in irrelevant_prefix))}
    encoder_signature = set(inspect.signature(encoder.forward).parameters)
    encoder_accepts_wildcard = 'kwargs' in encoder_signature or 'model_kwargs' in encoder_signature
    if not encoder_accepts_wildcard:
        encoder_kwargs = {argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature}
    model_input_name = model_input_name if model_input_name is not None else self.audio_encoder.main_input_name
    encoder_kwargs['return_dict'] = True
    if self.decoder.config.audio_channels == 1:
        encoder_kwargs[model_input_name] = input_values
        audio_encoder_outputs = encoder.encode(**encoder_kwargs)
        audio_codes = audio_encoder_outputs.audio_codes
        audio_scales = audio_encoder_outputs.audio_scales
        frames, bsz, codebooks, seq_len = audio_codes.shape
    else:
        if input_values.shape[1] != 2:
            raise ValueError(f'Expected stereo audio (2-channels) but example has {input_values.shape[1]} channel.')
        encoder_kwargs[model_input_name] = input_values[:, :1, :]
        audio_encoder_outputs_left = encoder.encode(**encoder_kwargs)
        audio_codes_left = audio_encoder_outputs_left.audio_codes
        audio_scales_left = audio_encoder_outputs_left.audio_scales
        encoder_kwargs[model_input_name] = input_values[:, 1:, :]
        audio_encoder_outputs_right = encoder.encode(**encoder_kwargs)
        audio_codes_right = audio_encoder_outputs_right.audio_codes
        audio_scales_right = audio_encoder_outputs_right.audio_scales
        frames, bsz, codebooks, seq_len = audio_codes_left.shape
        audio_codes = audio_codes_left.new_ones((frames, bsz, 2 * codebooks, seq_len))
        audio_codes[:, :, ::2, :] = audio_codes_left
        audio_codes[:, :, 1::2, :] = audio_codes_right
        if audio_scales_left != [None] or audio_scales_right != [None]:
            audio_scales = torch.stack([audio_scales_left, audio_scales_right], dim=1)
        else:
            audio_scales = [None] * bsz
    if frames != 1:
        raise ValueError(f'Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is disabled by setting `chunk_length=None` in the audio encoder.')
    decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)
    model_kwargs['decoder_input_ids'] = decoder_input_ids
    model_kwargs['audio_scales'] = audio_scales
    return model_kwargs