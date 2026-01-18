import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class GPTBigCodeDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        past_key_value_shape = (self.batch_size, self.sequence_length, self.hidden_size // self.num_attention_heads * 2)
        return [self.random_float_tensor(past_key_value_shape, framework=framework, dtype=float_dtype) for _ in range(self.num_layers)]