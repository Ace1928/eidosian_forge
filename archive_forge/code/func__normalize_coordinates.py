import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def _normalize_coordinates(target_size: int, coords: np.ndarray, original_size: Tuple[int, int], is_bounding_box=False) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the original image size in (height, width)
    format.
    """
    old_height, old_width = original_size
    scale = target_size * 1.0 / max(old_height, old_width)
    new_height, new_width = (old_height * scale, old_width * scale)
    new_width = int(new_width + 0.5)
    new_height = int(new_height + 0.5)
    coords = deepcopy(coords).astype(float)
    if is_bounding_box:
        coords = coords.reshape(-1, 2, 2)
    coords[..., 0] = coords[..., 0] * (new_width / old_width)
    coords[..., 1] = coords[..., 1] * (new_height / old_height)
    if is_bounding_box:
        coords = coords.reshape(-1, 4)
    return coords