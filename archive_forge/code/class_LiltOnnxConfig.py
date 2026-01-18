import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class LiltOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(allow_new=True, MAX_2D_POSITION_EMBEDDINGS='max_2d_position_embeddings')

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {'input_ids': {0: 'batch_size', 1: 'sequence_length'}, 'bbox': {0: 'batch_size', 1: 'sequence_length'}, 'attention_mask': {0: 'batch_size', 1: 'sequence_length'}}