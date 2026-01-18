import functools
from typing import Callable, Dict, Type, Union
from transformers import PretrainedConfig
class NormalizedTextAndVisionConfig(NormalizedTextConfig, NormalizedVisionConfig):
    TEXT_CONFIG = None
    VISION_CONFIG = None

    def __getattr__(self, attr_name):
        if self.TEXT_CONFIG is not None and attr_name.upper() in dir(NormalizedTextConfig):
            attr_name = f'{self.TEXT_CONFIG}.{attr_name}'
        elif self.VISION_CONFIG is not None and attr_name.upper() in dir(NormalizedVisionConfig):
            attr_name = f'{self.VISION_CONFIG}.{attr_name}'
        return super().__getattr__(attr_name)