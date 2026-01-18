import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class T5DummySeq2SeqPastKeyValuesGenerator(DummySeq2SeqPastKeyValuesGenerator):

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        encoder_shape = (self.batch_size, self.normalized_config.encoder_num_attention_heads, self.encoder_sequence_length, self.normalized_config.key_value_dim)
        decoder_shape = (self.batch_size, self.normalized_config.decoder_num_attention_heads, self.sequence_length, self.normalized_config.key_value_dim)
        return [(self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype), self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype), self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype), self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype)) for _ in range(self.normalized_config.decoder_num_layers)]