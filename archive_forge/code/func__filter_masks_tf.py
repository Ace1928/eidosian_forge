import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def _filter_masks_tf(self, masks, iou_scores, original_size, cropped_box_image, pred_iou_thresh=0.88, stability_score_thresh=0.95, mask_threshold=0, stability_score_offset=1):
    """
        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pad the predicted masks if necessary.

        Args:
            masks (`tf.Tensor`):
                Input masks.
            iou_scores (`tf.Tensor`):
                List of IoU scores.
            original_size (`Tuple[int,int]`):
                Size of the orginal image.
            cropped_box_image (`np.array`):
                The cropped image.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                The threshold for the iou scores.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                The threshold for the stability score.
            mask_threshold (`float`, *optional*, defaults to 0):
                The threshold for the predicted masks.
            stability_score_offset (`float`, *optional*, defaults to 1):
                The offset for the stability score used in the `_compute_stability_score` method.

        """
    requires_backends(self, ['tf'])
    original_height, original_width = original_size
    iou_scores = tf.reshape(iou_scores, [iou_scores.shape[0] * iou_scores.shape[1], iou_scores.shape[2:]])
    masks = tf.reshape(masks, [masks.shape[0] * masks.shape[1], masks.shape[2:]])
    if masks.shape[0] != iou_scores.shape[0]:
        raise ValueError('masks and iou_scores must have the same batch size.')
    batch_size = masks.shape[0]
    keep_mask = tf.ones(batch_size, dtype=tf.bool)
    if pred_iou_thresh > 0.0:
        keep_mask = keep_mask & (iou_scores > pred_iou_thresh)
    if stability_score_thresh > 0.0:
        stability_scores = _compute_stability_score_tf(masks, mask_threshold, stability_score_offset)
        keep_mask = keep_mask & (stability_scores > stability_score_thresh)
    scores = iou_scores[keep_mask]
    masks = masks[keep_mask]
    masks = masks > mask_threshold
    converted_boxes = _batched_mask_to_box_tf(masks)
    keep_mask = ~_is_box_near_crop_edge_tf(converted_boxes, cropped_box_image, [0, 0, original_width, original_height])
    scores = scores[keep_mask]
    masks = masks[keep_mask]
    converted_boxes = converted_boxes[keep_mask]
    masks = _pad_masks_tf(masks, cropped_box_image, original_height, original_width)
    masks = _mask_to_rle_tf(masks)
    return (masks, scores, converted_boxes)