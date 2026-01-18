import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class Data2VecAudioOnnxConfig(AudioOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig
    ATOL_FOR_VALIDATION = 0.0001