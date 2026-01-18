from ...utils import DummyTextInputGenerator, DummyVisionInputGenerator, logging
from .base import TFLiteConfig
class VisionTFLiteConfig(TFLiteConfig):
    """
    Handles vision architectures.
    """
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
    MANDATORY_AXES = ('batch_size', 'num_channels', 'width', 'height')