import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class PerceiverDummyInputGenerator(DummyVisionInputGenerator):

    def __init__(self, task: str, normalized_config: NormalizedVisionConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], num_channels: int=DEFAULT_DUMMY_SHAPES['num_channels'], width: int=DEFAULT_DUMMY_SHAPES['width'], height: int=DEFAULT_DUMMY_SHAPES['height'], **kwargs):
        super().__init__(task=task, normalized_config=normalized_config, batch_size=batch_size, num_channels=num_channels, width=width, height=height, **kwargs)
        from transformers.onnx.utils import get_preprocessor
        preprocessor = get_preprocessor(normalized_config._name_or_path)
        if preprocessor is not None and hasattr(preprocessor, 'size'):
            self.height = preprocessor.size.get('height', self.height)
            self.width = preprocessor.size.get('width', self.width)

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        input_ = super().generate(input_name=input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)
        return input_