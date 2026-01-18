import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummyTimestepInputGenerator(DummyInputGenerator):
    """
    Generates dummy time step inputs.
    """
    SUPPORTED_INPUT_NAMES = ('timestep', 'text_embeds', 'time_ids', 'timestep_cond')

    def __init__(self, task: str, normalized_config: NormalizedConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], random_batch_size_range: Optional[Tuple[int, int]]=None, **kwargs):
        self.task = task
        self.vocab_size = normalized_config.vocab_size
        self.text_encoder_projection_dim = normalized_config.text_encoder_projection_dim
        self.time_ids = 5 if normalized_config.requires_aesthetics_score else 6
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        self.time_cond_proj_dim = normalized_config.config.time_cond_proj_dim

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        if input_name == 'timestep':
            shape = [self.batch_size]
            return self.random_int_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=int_dtype)
        if input_name == 'text_embeds':
            dim = self.text_encoder_projection_dim
        elif input_name == 'timestep_cond':
            dim = self.time_cond_proj_dim
        else:
            dim = self.time_ids
        shape = [self.batch_size, dim]
        return self.random_float_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=float_dtype)