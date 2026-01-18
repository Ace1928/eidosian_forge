import json
import warnings
import numpy as np
from keras.src import activations
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.utils import file_utils
def _preprocess_tensor_input(x, data_format, mode):
    """Preprocesses a tensor encoding a batch of images.

    Args:
      x: Input tensor, 3D or 4D.
      data_format: Data format of the image tensor.
      mode: One of "caffe", "tf" or "torch".
        - caffe: will convert the images from RGB to BGR,
            then will zero-center each color channel with
            respect to the ImageNet dataset,
            without scaling.
        - tf: will scale pixels between -1 and 1,
            sample-wise.
        - torch: will scale pixels between 0 and 1 and then
            will normalize each channel with respect to the
            ImageNet dataset.

    Returns:
        Preprocessed tensor.
    """
    ndim = len(x.shape)
    if mode == 'tf':
        x /= 127.5
        x -= 1.0
        return x
    elif mode == 'torch':
        x /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == 'channels_first':
            if len(x.shape) == 3:
                x = ops.stack([x[i, ...] for i in (2, 1, 0)], axis=0)
            else:
                x = ops.stack([x[:, i, :] for i in (2, 1, 0)], axis=1)
        else:
            x = ops.stack([x[..., i] for i in (2, 1, 0)], axis=-1)
        mean = [103.939, 116.779, 123.68]
        std = None
    mean_tensor = ops.convert_to_tensor(-np.array(mean), dtype=x.dtype)
    if data_format == 'channels_first':
        mean_tensor = ops.reshape(mean_tensor, (1, 3) + (1,) * (ndim - 2))
    else:
        mean_tensor = ops.reshape(mean_tensor, (1,) * (ndim - 1) + (3,))
    x += mean_tensor
    if std is not None:
        std_tensor = ops.convert_to_tensor(np.array(std), dtype=x.dtype)
        if data_format == 'channels_first':
            std_tensor = ops.reshape(std_tensor, (-1, 1, 1))
        x /= std_tensor
    return x