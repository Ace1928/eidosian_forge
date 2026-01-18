import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class ASTDummyAudioInputGenerator(DummyAudioInputGenerator):

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        shape = [self.batch_size, self.normalized_config.max_length, self.normalized_config.num_mel_bins]
        if input_name == 'input_values':
            return self.random_float_tensor(shape, min_value=-1, max_value=1, framework=framework, dtype=float_dtype)
        return super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)