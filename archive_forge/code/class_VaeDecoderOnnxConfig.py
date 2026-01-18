import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class VaeDecoderOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 0.001
    DEFAULT_ONNX_OPSET = 14
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(num_channels='latent_channels', allow_new=True)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {'latent_sample': {0: 'batch_size', 1: 'num_channels_latent', 2: 'height_latent', 3: 'width_latent'}}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {'sample': {0: 'batch_size', 1: 'num_channels', 2: 'height', 3: 'width'}}