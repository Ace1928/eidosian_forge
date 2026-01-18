import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class SamOnnxConfig(OnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse('4.29.0.dev0')
    MIN_TORCH_VERSION = version.parse('2.0.99')
    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyPointsGenerator, DummyVisionEmbeddingsGenerator)
    DEFAULT_ONNX_OPSET = 13
    VARIANTS = {'monolith': 'All the SAM model components are exported as a single model.onnx.', 'split': 'The vision encoder is exported as a separate vision_encoder.onnx, and the prompt encoder and mask decoder are exported as a prompt_encoder_mask_decoder.onnx. This allows to encoder the image only once for multiple point queries.'}
    DEFAULT_VARIANT = 'split'

    def __init__(self, config: 'PretrainedConfig', task: str='feature-extraction', int_dtype: str='int64', float_dtype: str='fp32', variant: str='split', vision_encoder: Optional[bool]=None, preprocessors: Optional[List[Any]]=None, legacy: bool=False):
        super().__init__(config=config, task=task, int_dtype=int_dtype, float_dtype=float_dtype, preprocessors=preprocessors, legacy=legacy)
        self.variant = variant
        self.vision_encoder = vision_encoder
        self._normalized_config.ENCODER_NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig(self._config.vision_config)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self.variant == 'monolith':
            inputs = {'pixel_values': {0: 'batch_size'}, 'input_points': {0: 'batch_size', 1: 'point_batch_size', 2: 'nb_points_per_image'}, 'input_labels': {0: 'batch_size', 1: 'point_batch_size', 2: 'nb_points_per_image'}}
        elif self.vision_encoder:
            inputs = {'pixel_values': {0: 'batch_size'}}
        else:
            inputs = {'image_positional_embeddings': {0: 'batch_size'}, 'image_embeddings': {0: 'batch_size'}, 'input_points': {0: 'batch_size', 1: 'point_batch_size', 2: 'nb_points_per_image'}, 'input_labels': {0: 'batch_size', 1: 'point_batch_size', 2: 'nb_points_per_image'}}
        return inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self.variant == 'split' and self.vision_encoder:
            return {'image_embeddings': {0: 'batch_size'}, 'image_positional_embeddings': {0: 'batch_size'}}
        else:
            return {'iou_scores': {0: 'batch_size', 1: 'point_batch_size'}, 'pred_masks': {0: 'batch_size', 1: 'point_batch_size'}}

    def patch_model_for_export(self, model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Optional[Dict[str, Any]]=None) -> 'ModelPatcher':
        return SAMModelPatcher(self, model, model_kwargs=model_kwargs)