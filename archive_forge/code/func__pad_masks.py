import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def _pad_masks(masks, crop_box: List[int], orig_height: int, orig_width: int):
    left, top, right, bottom = crop_box
    if left == 0 and top == 0 and (right == orig_width) and (bottom == orig_height):
        return masks
    pad_x, pad_y = (orig_width - (right - left), orig_height - (bottom - top))
    pad = (left, pad_x - left, top, pad_y - top)
    return torch.nn.functional.pad(masks, pad, value=0)