from typing import List
from ...utils.normalized_config import NormalizedConfigManager
from .base import QuantizationApproach
from .config import TextEncoderTFliteConfig, VisionTFLiteConfig
class BertTFLiteConfig(TextEncoderTFliteConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class('bert')
    SUPPORTED_QUANTIZATION_APPROACHES = (QuantizationApproach.INT8_DYNAMIC, QuantizationApproach.INT8, QuantizationApproach.FP16)

    @property
    def inputs(self) -> List[str]:
        return ['input_ids', 'attention_mask', 'token_type_ids']