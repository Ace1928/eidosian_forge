import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
@staticmethod
def _infer_framework_from_input(input_) -> str:
    framework = None
    if is_torch_available() and isinstance(input_, torch.Tensor):
        framework = 'pt'
    elif is_tf_available() and isinstance(input_, tf.Tensor):
        framework = 'tf'
    elif isinstance(input_, np.ndarray):
        framework = 'np'
    else:
        raise RuntimeError(f'Could not infer the framework from {input_}')
    return framework