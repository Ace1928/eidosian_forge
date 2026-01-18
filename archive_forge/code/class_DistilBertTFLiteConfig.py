from typing import List
from ...utils.normalized_config import NormalizedConfigManager
from .base import QuantizationApproach
from .config import TextEncoderTFliteConfig, VisionTFLiteConfig
class DistilBertTFLiteConfig(BertTFLiteConfig):

    @property
    def inputs(self) -> List[str]:
        return ['input_ids', 'attention_mask']