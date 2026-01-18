import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummyVisionInputGenerator(DummyInputGenerator):
    """
    Generates dummy vision inputs.
    """
    SUPPORTED_INPUT_NAMES = ('pixel_values', 'pixel_mask', 'sample', 'latent_sample')

    def __init__(self, task: str, normalized_config: NormalizedVisionConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], num_channels: int=DEFAULT_DUMMY_SHAPES['num_channels'], width: int=DEFAULT_DUMMY_SHAPES['width'], height: int=DEFAULT_DUMMY_SHAPES['height'], **kwargs):
        self.task = task
        if normalized_config.has_attribute('num_channels'):
            self.num_channels = normalized_config.num_channels
        else:
            self.num_channels = num_channels
        if normalized_config.has_attribute('image_size'):
            self.image_size = normalized_config.image_size
        elif normalized_config.has_attribute('input_size'):
            input_size = normalized_config.input_size
            self.num_channels = input_size[0]
            self.image_size = input_size[1:]
        else:
            self.image_size = (height, width)
        if not isinstance(self.image_size, (tuple, list)):
            self.image_size = (self.image_size, self.image_size)
        self.batch_size = batch_size
        self.height, self.width = self.image_size

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        if input_name == 'pixel_mask':
            return self.random_int_tensor(shape=[self.batch_size, self.height, self.width], max_value=1, framework=framework, dtype=int_dtype)
        else:
            return self.random_float_tensor(shape=[self.batch_size, self.num_channels, self.height, self.width], framework=framework, dtype=float_dtype)