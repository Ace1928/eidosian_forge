import json
import warnings
import numpy as np
from keras.src import activations
from keras.src import backend
from keras.src.utils import data_utils
from tensorflow.python.util.tf_export import keras_export
def _preprocess_symbolic_input(x, data_format, mode):
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
            if backend.ndim(x) == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None
    mean_tensor = backend.constant(-np.array(mean))
    if backend.dtype(x) != backend.dtype(mean_tensor):
        x = backend.bias_add(x, backend.cast(mean_tensor, backend.dtype(x)), data_format=data_format)
    else:
        x = backend.bias_add(x, mean_tensor, data_format)
    if std is not None:
        std_tensor = backend.constant(np.array(std), dtype=backend.dtype(x))
        if data_format == 'channels_first':
            std_tensor = backend.reshape(std_tensor, (-1, 1, 1))
        x /= std_tensor
    return x