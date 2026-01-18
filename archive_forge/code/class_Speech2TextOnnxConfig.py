import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class Speech2TextOnnxConfig(AudioToTextOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(decoder_num_layers='decoder_layers', num_layers='decoder_layers', input_features_per_channel='input_feat_per_channel', allow_new=True)
    DUMMY_INPUT_GENERATOR_CLASSES = (Speech2TextDummyAudioInputGenerator,) + AudioToTextOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES[1:] + (DummyTextInputGenerator,)
    ATOL_FOR_VALIDATION = 0.0001

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}
        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs['input_features'] = {0: 'batch_size', 1: 'feature_size', 2: 'encoder_sequence_length'}
            common_inputs['attention_mask'] = {0: 'batch_size', 1: 'encoder_sequence_length'}
        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                common_inputs['decoder_input_ids'] = {0: 'batch_size'}
            else:
                common_inputs['decoder_input_ids'] = {0: 'batch_size', 1: 'decoder_sequence_length'}
            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction='inputs')
        if self._behavior is ConfigBehavior.DECODER:
            common_inputs['encoder_outputs'] = {0: 'batch_size', 1: f'encoder_sequence_length / {2 * self._config.num_conv_layers}'}
        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs
        if self._behavior is ConfigBehavior.ENCODER:
            common_outputs['last_hidden_state'][1] = f'{common_outputs['last_hidden_state'][1]} / {2 * self._config.num_conv_layers}'
        return common_outputs