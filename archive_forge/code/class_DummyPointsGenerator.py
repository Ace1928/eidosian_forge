import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummyPointsGenerator(DummyInputGenerator):
    """
    Generates dummy time step inputs.
    """
    SUPPORTED_INPUT_NAMES = ('input_points', 'input_labels')

    def __init__(self, task: str, normalized_config: NormalizedConfig, batch_size: int=DEFAULT_DUMMY_SHAPES['batch_size'], point_batch_size: int=DEFAULT_DUMMY_SHAPES['point_batch_size'], nb_points_per_image: int=DEFAULT_DUMMY_SHAPES['nb_points_per_image'], **kwargs):
        self.task = task
        self.batch_size = batch_size
        self.point_batch_size = point_batch_size
        self.nb_points_per_image = nb_points_per_image

    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        if input_name == 'input_points':
            shape = [self.batch_size, self.point_batch_size, self.nb_points_per_image, 2]
            return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)
        else:
            shape = [self.batch_size, self.point_batch_size, self.nb_points_per_image]
            return self.random_int_tensor(shape, min_value=0, max_value=1, framework=framework, dtype=int_dtype)