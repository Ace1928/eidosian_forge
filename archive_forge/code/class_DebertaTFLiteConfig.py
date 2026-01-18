from typing import List
from ...utils.normalized_config import NormalizedConfigManager
from .base import QuantizationApproach
from .config import TextEncoderTFliteConfig, VisionTFLiteConfig
class DebertaTFLiteConfig(BertTFLiteConfig):
    SUPPORTED_QUANTIZATION_APPROACHES = (QuantizationApproach.INT8_DYNAMIC, QuantizationApproach.FP16)

    @property
    def inputs(self) -> List[str]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            common_inputs.pop(-1)
        return common_inputs