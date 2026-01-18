import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def _generate_crop_images(crop_boxes, image, points_grid, layer_idxs, target_size, original_size, input_data_format=None):
    """
    Takes as an input bounding boxes that are used to crop the image. Based in the crops, the corresponding points are
    also passed.
    """
    cropped_images = []
    total_points_per_crop = []
    for i, crop_box in enumerate(crop_boxes):
        left, top, right, bottom = crop_box
        channel_dim = infer_channel_dimension_format(image, input_data_format)
        if channel_dim == ChannelDimension.LAST:
            cropped_im = image[top:bottom, left:right, :]
        else:
            cropped_im = image[:, top:bottom, left:right]
        cropped_images.append(cropped_im)
        cropped_im_size = get_image_size(cropped_im, channel_dim)
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points = points_grid[layer_idxs[i]] * points_scale
        normalized_points = _normalize_coordinates(target_size, points, original_size)
        total_points_per_crop.append(normalized_points)
    return (cropped_images, total_points_per_crop)