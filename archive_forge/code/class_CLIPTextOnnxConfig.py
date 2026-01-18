import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class CLIPTextOnnxConfig(CLIPTextWithProjectionOnnxConfig):

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = {'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}, 'pooler_output': {0: 'batch_size'}}
        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs[f'hidden_states.{i}'] = {0: 'batch_size', 1: 'sequence_length'}
        return common_outputs

    def generate_dummy_inputs(self, framework: str='pt', **kwargs):
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        if framework == 'pt':
            import torch
            dummy_inputs['input_ids'] = dummy_inputs['input_ids'].to(dtype=torch.int32)
        return dummy_inputs