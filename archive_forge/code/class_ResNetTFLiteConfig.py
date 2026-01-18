from typing import List
from ...utils.normalized_config import NormalizedConfigManager
from .base import QuantizationApproach
from .config import TextEncoderTFliteConfig, VisionTFLiteConfig
class ResNetTFLiteConfig(VisionTFLiteConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class('resnet')

    @property
    def inputs(self) -> List[str]:
        return ['pixel_values']