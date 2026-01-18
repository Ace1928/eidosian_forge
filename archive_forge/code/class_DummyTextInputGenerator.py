import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummyTextInputGenerator(DummyInputGenerator):
    """
    Generates dummy encoder text inputs.
    """
    SUPPORTED_INPUT_NAMES = ('input_ids', 'attention_mask', 'encoder_attention_mask', 'token_type_ids', 'position_ids')

    def __init__(self, task: str, normalized_config: NormalizedTextConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], sequence_length: int=DEFAULT_DUMMY_SHAPES['sequence_length'], num_choices: int=DEFAULT_DUMMY_SHAPES['num_choices'], random_batch_size_range: Optional[Tuple[int, int]]=None, random_sequence_length_range: Optional[Tuple[int, int]]=None, random_num_choices_range: Optional[Tuple[int, int]]=None, padding_side: str='right', **kwargs):
        self.task = task
        self.vocab_size = normalized_config.vocab_size
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length
        if random_num_choices_range:
            low, high = random_num_choices_range
            self.num_choices = random.randint(low, high)
        else:
            self.num_choices = num_choices
        self.padding_side = padding_side

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        min_value = 0
        max_value = 2 if input_name != 'input_ids' else self.vocab_size
        shape = [self.batch_size, self.sequence_length]
        if self.task == 'multiple-choice':
            shape = [self.batch_size, self.num_choices, self.sequence_length]
        if 'mask' in input_name:
            return self.random_mask_tensor(shape, padding_side=self.padding_side, framework=framework, dtype=int_dtype)
        else:
            return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)