import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummyLabelsGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ('labels', 'start_positions', 'end_positions')

    def __init__(self, task: str, normalized_config: NormalizedConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], random_batch_size_range: Optional[Tuple[int, int]]=None, **kwargs):
        self.task = task
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        self.sequence_length = kwargs.get('sequence_length', None)
        self.num_labels = kwargs.get('num_labels', None)

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        max_value = self.num_labels if self.num_labels is not None else 0
        if self.sequence_length is None:
            shape = [self.batch_size]
        else:
            shape = [self.batch_size, self.sequence_length]
        return self.random_int_tensor(shape, max_value=max_value, framework=framework, dtype=int_dtype)