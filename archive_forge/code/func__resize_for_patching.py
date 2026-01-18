import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict, select_best_resolution
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def _resize_for_patching(self, image: np.array, target_resolution: tuple, resample, input_data_format: ChannelDimension) -> np.array:
    """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image (np.array):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            np.array: The resized and padded image.
        """
    new_height, new_width = _get_patch_output_size(image, target_resolution, input_data_format)
    resized_image = resize(image, (new_height, new_width), resample=resample, input_data_format=input_data_format)
    return resized_image