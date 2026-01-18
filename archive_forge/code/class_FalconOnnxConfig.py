import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class FalconOnnxConfig(TextDecoderOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse('4.35.99')
    DUMMY_INPUT_GENERATOR_CLASSES = (FalconDummyPastKeyValuesGenerator,) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DEFAULT_ONNX_OPSET = 14
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_PKV_GENERATOR_CLASS = FalconDummyPastKeyValuesGenerator

    def __init__(self, config: 'PretrainedConfig', task: str='feature-extraction', int_dtype: str='int64', float_dtype: str='fp32', use_past: bool=False, use_past_in_inputs: bool=False, preprocessors: Optional[List[Any]]=None, legacy: bool=False):
        super().__init__(config=config, task=task, int_dtype=int_dtype, float_dtype=float_dtype, use_past=use_past, use_past_in_inputs=use_past_in_inputs, preprocessors=preprocessors, legacy=legacy)
        self._normalized_config.num_kv_heads = self._normalized_config.num_kv_heads if self._normalized_config.new_decoder_architecture or not self._normalized_config.multi_query else 1

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs
        if not self.legacy and (not self._config.alibi) and (self.task in ['text-generation', 'feature-extraction']):
            common_inputs['position_ids'] = {0: 'batch_size', 1: 'sequence_length'}
        return common_inputs

    def patch_model_for_export(self, model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Optional[Dict[str, Any]]=None) -> 'ModelPatcher':
        return FalconModelPatcher(self, model, model_kwargs=model_kwargs)