import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummySeq2SeqPastKeyValuesGenerator(DummyInputGenerator):
    """
    Generates dummy past_key_values inputs for seq2seq architectures.
    """
    SUPPORTED_INPUT_NAMES = ('past_key_values',)

    def __init__(self, task: str, normalized_config: NormalizedSeq2SeqConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], sequence_length: int=DEFAULT_DUMMY_SHAPES['sequence_length'], encoder_sequence_length: Optional[int]=None, random_batch_size_range: Optional[Tuple[int, int]]=None, random_sequence_length_range: Optional[Tuple[int, int]]=None, **kwargs):
        self.normalized_config = normalized_config
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
        self.encoder_sequence_length = self.sequence_length if encoder_sequence_length is None else encoder_sequence_length

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        encoder_shape = (self.batch_size, self.normalized_config.encoder_num_attention_heads, self.encoder_sequence_length, self.normalized_config.hidden_size // self.normalized_config.encoder_num_attention_heads)
        decoder_shape = (self.batch_size, self.normalized_config.decoder_num_attention_heads, self.sequence_length, self.normalized_config.hidden_size // self.normalized_config.decoder_num_attention_heads)
        return [(self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype), self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype), self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype), self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype)) for _ in range(self.normalized_config.decoder_num_layers)]