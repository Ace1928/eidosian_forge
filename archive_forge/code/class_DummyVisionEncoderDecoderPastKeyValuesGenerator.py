import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummyVisionEncoderDecoderPastKeyValuesGenerator(DummySeq2SeqPastKeyValuesGenerator):

    def __init__(self, task: str, normalized_config: NormalizedSeq2SeqConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], sequence_length: int=DEFAULT_DUMMY_SHAPES['sequence_length'], encoder_sequence_length: Optional[int]=None, random_batch_size_range: Optional[Tuple[int, int]]=None, random_sequence_length_range: Optional[Tuple[int, int]]=None, **kwargs):
        super().__init__(task=task, normalized_config=normalized_config, batch_size=batch_size, sequence_length=sequence_length, encoder_sequence_length=encoder_sequence_length, random_batch_size_range=random_batch_size_range, random_sequence_length_range=random_sequence_length_range, **kwargs)
        if normalized_config.model_type == 'trocr':
            image_size = normalized_config.encoder.image_size
            patch_size = normalized_config.encoder.patch_size
            self.encoder_sequence_length = (image_size // patch_size) ** 2 + 1
        if isinstance(normalized_config.DECODER_NORMALIZED_CONFIG_CLASS, NormalizedSeq2SeqConfig):
            self.num_layers = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.decoder_num_layers
            self.use_cross_attention = True
        else:
            self.num_layers = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_layers
            self.use_cross_attention = False

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        decoder_hidden_size = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.hidden_size
        decoder_num_attention_heads = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_attention_heads
        decoder_shape = (self.batch_size, decoder_num_attention_heads, self.sequence_length, decoder_hidden_size // decoder_num_attention_heads)
        if not self.use_cross_attention:
            return [(self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype), self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype)) for _ in range(self.num_layers)]
        else:
            encoder_hidden_size = decoder_hidden_size
            encoder_num_attention_heads = decoder_num_attention_heads
            encoder_shape = (self.batch_size, encoder_num_attention_heads, self.encoder_sequence_length, encoder_hidden_size // encoder_num_attention_heads)
            return [(self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype), self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype), self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype), self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype)) for _ in range(self.num_layers)]