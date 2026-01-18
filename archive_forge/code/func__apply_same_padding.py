import torch
import torch.nn.functional as tnn
import tree
from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import (
from keras.src.backend.config import epsilon
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.numpy import expand_dims
from keras.src.backend.torch.numpy import maximum
from keras.src.backend.torch.numpy import where
from keras.src.utils.argument_validation import standardize_tuple
def _apply_same_padding(inputs, kernel_size, strides, operation_type, dilation_rate=1):
    """Apply same padding to the input tensor.

    This function will evaluate if the padding value is compatible with torch
    functions. To avoid calling `pad()` as much as possible, which may cause
    performance or memory issues, when compatible, it does not apply the padding
    to the tensor, but returns the input tensor and the padding value to pass to
    the torch functions. If not compatible, it returns the padded tensor and 0
    as the padding value.

    Returns:
        tensor: A padded tensor or the inputs.
        padding: The padding value, ready to pass to the torch functions.
    """
    spatial_shape = inputs.shape[2:]
    num_spatial_dims = len(spatial_shape)
    padding = ()
    for i in range(num_spatial_dims):
        if operation_type == 'pooling':
            padding_size = _compute_padding_length(spatial_shape[i], kernel_size[i], strides[i])
            mode = 'replicate'
        else:
            dilation_rate = standardize_tuple(dilation_rate, num_spatial_dims, 'dilation_rate')
            padding_size = _compute_padding_length(spatial_shape[i], kernel_size[i], strides[i], dilation_rate[i])
            mode = 'constant'
        padding = (padding_size,) + padding
    if all([left == right for left, right in padding]):
        return (inputs, [left for left, _ in padding])
    flattened_padding = tuple((value for left_and_right in padding for value in left_and_right))
    return (tnn.pad(inputs, pad=flattened_padding, mode=mode), 0)