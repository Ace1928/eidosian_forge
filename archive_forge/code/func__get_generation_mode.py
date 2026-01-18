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
def _get_generation_mode(self, generation_config: GenerationConfig, assistant_model: Optional['PreTrainedModel']) -> GenerationMode:
    """
        Returns the generation mode triggered by a [`GenerationConfig`] instance.
        """
    if generation_config.constraints is not None or generation_config.force_words_ids is not None:
        generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
    elif generation_config.num_beams == 1:
        if generation_config.do_sample is False:
            if generation_config.top_k is not None and generation_config.top_k > 1 and (generation_config.penalty_alpha is not None) and (generation_config.penalty_alpha > 0):
                generation_mode = GenerationMode.CONTRASTIVE_SEARCH
            else:
                generation_mode = GenerationMode.GREEDY_SEARCH
        else:
            generation_mode = GenerationMode.SAMPLE
    elif generation_config.num_beam_groups > 1:
        generation_mode = GenerationMode.GROUP_BEAM_SEARCH
    elif generation_config.do_sample is True:
        generation_mode = GenerationMode.BEAM_SAMPLE
    else:
        generation_mode = GenerationMode.BEAM_SEARCH
    if assistant_model is not None or generation_config.prompt_lookup_num_tokens is not None:
        if generation_mode in ('greedy_search', 'sample'):
            generation_mode = GenerationMode.ASSISTED_GENERATION
        else:
            raise ValueError("You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate is only supported with Greedy Search and Sample.")
    return generation_mode