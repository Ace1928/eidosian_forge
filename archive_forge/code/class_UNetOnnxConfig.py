import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class UNetOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 0.001
    DEFAULT_ONNX_OPSET = 14
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(image_size='sample_size', num_channels='in_channels', hidden_size='cross_attention_dim', vocab_size='norm_num_groups', allow_new=True)
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyTimestepInputGenerator, DummySeq2SeqDecoderTextInputGenerator)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {'sample': {0: 'batch_size', 1: 'num_channels', 2: 'height', 3: 'width'}, 'timestep': {0: 'steps'}, 'encoder_hidden_states': {0: 'batch_size', 1: 'sequence_length'}}
        if getattr(self._normalized_config, 'addition_embed_type', None) == 'text_time':
            common_inputs['text_embeds'] = {0: 'batch_size'}
            common_inputs['time_ids'] = {0: 'batch_size'}
        if getattr(self._normalized_config, 'time_cond_proj_dim', None) is not None:
            common_inputs['timestep_cond'] = {0: 'batch_size'}
        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {'out_sample': {0: 'batch_size', 1: 'num_channels', 2: 'height', 3: 'width'}}

    @property
    def torch_to_onnx_output_map(self) -> Dict[str, str]:
        return {'sample': 'out_sample'}

    def generate_dummy_inputs(self, framework: str='pt', **kwargs):
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        dummy_inputs['encoder_hidden_states'] = dummy_inputs['encoder_hidden_states'][0]
        if getattr(self._normalized_config, 'addition_embed_type', None) == 'text_time':
            dummy_inputs['added_cond_kwargs'] = {'text_embeds': dummy_inputs.pop('text_embeds'), 'time_ids': dummy_inputs.pop('time_ids')}
        return dummy_inputs

    def ordered_inputs(self, model) -> Dict[str, Dict[int, str]]:
        inputs = super().ordered_inputs(model=model)
        if getattr(self._normalized_config, 'addition_embed_type', None) == 'text_time':
            inputs['text_embeds'] = self.inputs['text_embeds']
            inputs['time_ids'] = self.inputs['time_ids']
        return inputs