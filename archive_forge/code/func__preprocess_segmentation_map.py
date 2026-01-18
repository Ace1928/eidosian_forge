import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging
def _preprocess_segmentation_map(self, segmentation_map: ImageInput, do_resize: bool=None, size: Dict[str, int]=None, resample: PILImageResampling=None, do_center_crop: bool=None, crop_size: Dict[str, int]=None, do_reduce_labels: bool=None, input_data_format: Optional[Union[str, ChannelDimension]]=None):
    """Preprocesses a single segmentation map."""
    segmentation_map = to_numpy_array(segmentation_map)
    if segmentation_map.ndim == 2:
        segmentation_map = segmentation_map[None, ...]
        added_dimension = True
        input_data_format = ChannelDimension.FIRST
    else:
        added_dimension = False
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
    segmentation_map = self._preprocess(image=segmentation_map, do_reduce_labels=do_reduce_labels, do_resize=do_resize, resample=resample, size=size, do_center_crop=do_center_crop, crop_size=crop_size, do_normalize=False, do_rescale=False, input_data_format=ChannelDimension.FIRST)
    if added_dimension:
        segmentation_map = np.squeeze(segmentation_map, axis=0)
    segmentation_map = segmentation_map.astype(np.int64)
    return segmentation_map