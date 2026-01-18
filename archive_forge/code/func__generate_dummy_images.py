import copy
import dataclasses
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import numpy as np
from packaging import version
from ..utils import TensorType, is_torch_available, is_vision_available, logging
from .utils import ParameterFormat, compute_effective_axis_dimension, compute_serialized_parameters_size
def _generate_dummy_images(self, batch_size: int=2, num_channels: int=3, image_height: int=40, image_width: int=40):
    images = []
    for _ in range(batch_size):
        data = np.random.rand(image_height, image_width, num_channels) * 255
        images.append(Image.fromarray(data.astype('uint8')).convert('RGB'))
    return images