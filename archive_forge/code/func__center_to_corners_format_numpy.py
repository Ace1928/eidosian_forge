import warnings
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
from .image_utils import (
from .utils import ExplicitEnum, TensorType, is_jax_tensor, is_tf_tensor, is_torch_tensor
from .utils.import_utils import (
def _center_to_corners_format_numpy(bboxes_center: np.ndarray) -> np.ndarray:
    center_x, center_y, width, height = bboxes_center.T
    bboxes_corners = np.stack([center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height], axis=-1)
    return bboxes_corners