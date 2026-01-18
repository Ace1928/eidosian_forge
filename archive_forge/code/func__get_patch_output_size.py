import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict, select_best_resolution
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def _get_patch_output_size(image, target_resolution, input_data_format):
    original_height, original_width = get_image_size(image, channel_dim=input_data_format)
    target_height, target_width = target_resolution
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)
    return (new_height, new_width)