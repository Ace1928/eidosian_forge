import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def _preprocess_image(self, image: ImageInput, do_resize: bool=None, size: Dict[str, int]=None, resample: PILImageResampling=None, do_rescale: bool=None, rescale_factor: float=None, do_normalize: bool=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
    """Preprocesses a single image."""
    image = to_numpy_array(image)
    if is_scaled_image(image) and do_rescale:
        logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    image = self._preprocess(image=image, do_resize=do_resize, size=size, resample=resample, do_rescale=do_rescale, rescale_factor=rescale_factor, do_normalize=do_normalize, image_mean=image_mean, image_std=image_std, input_data_format=input_data_format)
    if data_format is not None:
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
    return image