import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def _post_process_masks_tf(self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None):
    """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`tf.Tensor`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`tf.Tensor`):
                The original size of the images before resizing for input to the model, in (height, width) format.
            reshaped_input_sizes (`tf.Tensor`):
                The size of the image input to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.
        Returns:
            (`tf.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width) is
            given by original_size.
        """
    requires_backends(self, ['tf'])
    pad_size = self.pad_size if pad_size is None else pad_size
    target_image_size = (pad_size['height'], pad_size['width'])
    output_masks = []
    for i, original_size in enumerate(original_sizes):
        mask = tf.transpose(masks[i], perm=[0, 2, 3, 1])
        interpolated_mask = tf.image.resize(mask, target_image_size, method='bilinear')
        interpolated_mask = interpolated_mask[:, :reshaped_input_sizes[i][0], :reshaped_input_sizes[i][1], :]
        interpolated_mask = tf.image.resize(interpolated_mask, original_size, method='bilinear')
        if binarize:
            interpolated_mask = interpolated_mask > mask_threshold
        output_masks.append(tf.transpose(interpolated_mask, perm=[0, 3, 1, 2]))
    return output_masks