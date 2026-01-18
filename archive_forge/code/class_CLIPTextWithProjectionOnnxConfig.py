import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class CLIPTextWithProjectionOnnxConfig(TextEncoderOnnxConfig):
    ATOL_FOR_VALIDATION = 0.001
    DEFAULT_ONNX_OPSET = 14
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(vocab_size='vocab_size', sequence_length='max_position_embeddings', num_layers='num_hidden_layers', allow_new=True)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {'input_ids': {0: 'batch_size', 1: 'sequence_length'}}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = {'text_embeds': {0: 'batch_size', 1: 'sequence_length'}, 'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}}
        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs[f'hidden_states.{i}'] = {0: 'batch_size', 1: 'sequence_length'}
        return common_outputs