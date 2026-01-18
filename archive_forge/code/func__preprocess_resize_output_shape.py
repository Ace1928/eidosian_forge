import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def _preprocess_resize_output_shape(image, output_shape):
    """Validate resize output shape according to input image.

    Args:
        image (`np.ndarray`):
         Image to be resized.
        output_shape (`iterable`):
            Size of the generated output image `(rows, cols[, ...][, dim])`. If `dim` is not provided, the number of
            channels is preserved.

    Returns
        image (`np.ndarray):
            The input image, but with additional singleton dimensions appended in the case where `len(output_shape) >
            input.ndim`.
        output_shape (`Tuple`):
            The output shape converted to tuple.

    Raises ------ ValueError:
        If output_shape length is smaller than the image number of dimensions.

    Notes ----- The input image is reshaped if its number of dimensions is not equal to output_shape_length.

    """
    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)
    input_shape = image.shape
    if output_ndim > image.ndim:
        input_shape += (1,) * (output_ndim - image.ndim)
        image = np.reshape(image, input_shape)
    elif output_ndim == image.ndim - 1:
        output_shape = output_shape + (image.shape[-1],)
    elif output_ndim < image.ndim:
        raise ValueError('output_shape length cannot be smaller than the image number of dimensions')
    return (image, output_shape)