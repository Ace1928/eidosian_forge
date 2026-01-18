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
class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """
    CONTRASTIVE_SEARCH = 'contrastive_search'
    GREEDY_SEARCH = 'greedy_search'
    SAMPLE = 'sample'
    ASSISTED_GENERATION = 'assisted_generation'
    BEAM_SEARCH = 'beam_search'
    BEAM_SAMPLE = 'beam_sample'
    CONSTRAINED_BEAM_SEARCH = 'constrained_beam_search'
    GROUP_BEAM_SEARCH = 'group_beam_search'