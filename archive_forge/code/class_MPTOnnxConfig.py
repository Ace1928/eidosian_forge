import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class MPTOnnxConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_attention_heads='n_heads', hidden_size='d_model', num_layers='n_layers')