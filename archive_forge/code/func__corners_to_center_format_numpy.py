import warnings
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
from .image_utils import (
from .utils import ExplicitEnum, TensorType, is_jax_tensor, is_tf_tensor, is_torch_tensor
from .utils.import_utils import (
def _corners_to_center_format_numpy(bboxes_corners: np.ndarray) -> np.ndarray:
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bboxes_corners.T
    bboxes_center = np.stack([(top_left_x + bottom_right_x) / 2, (top_left_y + bottom_right_y) / 2, bottom_right_x - top_left_x, bottom_right_y - top_left_y], axis=-1)
    return bboxes_center