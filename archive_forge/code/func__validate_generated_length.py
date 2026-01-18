import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn
from ..cache_utils import Cache, DynamicCache, StaticCache
from ..integrations.deepspeed import is_deepspeed_zero3_enabled
from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from ..models.auto import (
from ..utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
from .configuration_utils import GenerationConfig
from .logits_process import (
from .stopping_criteria import (
def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
    """Performs validation related to the resulting generated length"""
    if has_default_max_length and generation_config.max_new_tokens is None and (generation_config.max_length == 20):
        warnings.warn(f'Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.', UserWarning)
    if input_ids_length >= generation_config.max_length:
        input_ids_string = 'decoder_input_ids' if self.config.is_encoder_decoder else 'input_ids'
        raise ValueError(f'Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to {generation_config.max_length}. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.')
    min_length_error_suffix = ' Generation will stop at the defined maximum length. You should decrease the minimum length and/or increase the maximum length.'
    if has_default_max_length:
        min_length_error_suffix += f' Note that `max_length` is set to {generation_config.max_length}, its default value.'
    if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
        warnings.warn(f'Unfeasible length constraints: `min_length` ({generation_config.min_length}) is larger than the maximum possible length ({generation_config.max_length}).' + min_length_error_suffix, UserWarning)
    if generation_config.min_new_tokens is not None:
        min_length = generation_config.min_new_tokens + input_ids_length
        if min_length > generation_config.max_length:
            warnings.warn(f'Unfeasible length constraints: `min_new_tokens` ({generation_config.min_new_tokens}), when added to the prompt length ({input_ids_length}), is larger than the maximum possible length ({generation_config.max_length}).' + min_length_error_suffix, UserWarning)