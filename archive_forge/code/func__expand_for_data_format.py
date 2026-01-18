import warnings
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
from .image_utils import (
from .utils import ExplicitEnum, TensorType, is_jax_tensor, is_tf_tensor, is_torch_tensor
from .utils.import_utils import (
def _expand_for_data_format(values):
    """
        Convert values to be in the format expected by np.pad based on the data format.
        """
    if isinstance(values, (int, float)):
        values = ((values, values), (values, values))
    elif isinstance(values, tuple) and len(values) == 1:
        values = ((values[0], values[0]), (values[0], values[0]))
    elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], int):
        values = (values, values)
    elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], tuple):
        values = values
    else:
        raise ValueError(f'Unsupported format: {values}')
    values = ((0, 0), *values) if input_data_format == ChannelDimension.FIRST else (*values, (0, 0))
    values = (0, *values) if image.ndim == 4 else values
    return values