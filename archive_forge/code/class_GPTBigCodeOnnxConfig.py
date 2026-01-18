import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class GPTBigCodeOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (GPTBigCodeDummyPastKeyValuesGenerator,) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DEFAULT_ONNX_OPSET = 14
    DUMMY_PKV_GENERATOR_CLASS = GPTBigCodeDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class('gpt_bigcode')

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ['inputs', 'outputs']:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')
        if direction == 'inputs':
            decoder_sequence_name = 'past_sequence_length'
            name = 'past_key_values'
        else:
            decoder_sequence_name = 'past_sequence_length + 1'
            name = 'present'
        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f'{name}.{i}.key_value'] = {0: 'batch_size', 1: decoder_sequence_name}

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        flattened_output[f'{name}.{idx}.key_value'] = t