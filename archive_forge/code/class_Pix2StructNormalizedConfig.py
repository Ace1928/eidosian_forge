import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class Pix2StructNormalizedConfig(NormalizedSeq2SeqConfig):
    ENCODER_NUM_LAYERS = 'vision_config.num_hidden_layers'
    DECODER_NUM_LAYERS = 'text_config.num_layers'
    ENCODER_NUM_ATTENTION_HEADS = 'vision_config.num_attention_heads'
    DECODER_NUM_ATTENTION_HEADS = 'text_config.num_heads'
    HIDDEN_SIZE = 'text_config.hidden_size'
    VOCAB_SIZE = 'text_config.vocab_size'