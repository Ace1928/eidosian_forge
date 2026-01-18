import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummyPix2StructInputGenerator(DummyInputGenerator):
    """
    Generates dummy time step inputs.
    """
    SUPPORTED_INPUT_NAMES = ('flattened_patches',)

    def __init__(self, task: str, normalized_config: NormalizedConfig, preprocessors: List[Any], batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], num_channels: int=DEFAULT_DUMMY_SHAPES['num_channels'], **kwargs):
        self.task = task
        self.batch_size = batch_size
        patch_height = preprocessors[1].image_processor.patch_size['height']
        patch_width = preprocessors[1].image_processor.patch_size['width']
        self.flattened_patch_size = 2 + patch_height * patch_width * num_channels
        self.max_patches = preprocessors[1].image_processor.max_patches

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        shape = [self.batch_size, self.max_patches, self.flattened_patch_size]
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)