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
from ..utils import ModelOutput, is_accelerate_available, is_torchdynamo_compiling, logging
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
from .configuration_utils import GenerationConfig, GenerationMode
from .logits_process import (
from .stopping_criteria import (
def _prepare_generation_config(self, generation_config: GenerationConfig, **kwargs: Dict) -> Tuple[GenerationConfig, Dict]:
    """
        Prepares the base generation config, then applies any generation configuration options from kwargs.
        """
    if generation_config is None:
        if not is_torchdynamo_compiling() and self.generation_config._from_model_config and (self.generation_config._original_object_hash == hash(self.generation_config)) and self.config._has_non_default_generation_parameters():
            new_generation_config = GenerationConfig.from_model_config(self.config)
            if new_generation_config != self.generation_config:
                warnings.warn('You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. Please use and modify the model generation configuration (see https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )')
                self.generation_config = new_generation_config
        generation_config = self.generation_config
    if is_torchdynamo_compiling():
        model_kwargs = kwargs
        generate_attributes_in_kwargs = [key for key, value in kwargs.items() if getattr(generation_config, key, None) != value]
        if len(generate_attributes_in_kwargs) > 0:
            raise ValueError(f'`torch.compile` exception: all generation configuration attributes must be passed within a `generation_config` instance passed to `generate` (found: {generate_attributes_in_kwargs}).')
    else:
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
    return (generation_config, model_kwargs)