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
def _pad_image(self, image: np.ndarray, output_size: Tuple[int, int], constant_values: Union[float, Iterable[float]]=0, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
    """
        Pad an image with zeros to the given size.
        """
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    output_height, output_width = output_size
    pad_bottom = output_height - input_height
    pad_right = output_width - input_width
    padding = ((0, pad_bottom), (0, pad_right))
    padded_image = pad(image, padding, mode=PaddingMode.CONSTANT, constant_values=constant_values, data_format=data_format, input_data_format=input_data_format)
    return padded_image