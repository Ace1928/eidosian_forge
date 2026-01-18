import functools
from typing import Callable, Dict, Type, Union
from transformers import PretrainedConfig
class NormalizedEncoderDecoderConfig(NormalizedConfig):
    ENCODER_NORMALIZED_CONFIG_CLASS = None
    DECODER_NORMALIZED_CONFIG_CLASS = None

    def __getattr__(self, attr_name):
        if self.ENCODER_NORMALIZED_CONFIG_CLASS is not None and attr_name.upper() in dir(self.ENCODER_NORMALIZED_CONFIG_CLASS):
            return self.ENCODER_NORMALIZED_CONFIG_CLASS.__getattr__(attr_name)
        if self.DECODER_NORMALIZED_CONFIG_CLASS is not None and attr_name.upper() in dir(self.DECODER_NORMALIZED_CONFIG_CLASS):
            return self.DECODER_NORMALIZED_CONFIG_CLASS.__getattr__(attr_name)
        return super().__getattr__(attr_name)