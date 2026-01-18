from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class MaxPoolLayer(Layer):
    """
    2D max-pooling network layer.

    Parameters
    ----------
    size : tuple
        The size of the pooling region in each dimension.
    stride : tuple, optional
        The strides between successive pooling regions in each dimension.
        If None `stride` = `size`.

    """

    def __init__(self, size, stride=None):
        self.size = size
        if stride is None:
            stride = size
        self.stride = stride

    def activate(self, data, **kwargs):
        """
        Activate the layer.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Max pooled data.

        """
        from scipy.ndimage.filters import maximum_filter
        slice_dim_1 = slice(self.size[0] // 2, None, self.stride[0])
        slice_dim_2 = slice(self.size[1] // 2, None, self.stride[1])
        data = [maximum_filter(data[:, :, c], self.size, mode='constant')[slice_dim_1, slice_dim_2] for c in range(data.shape[2])]
        return np.dstack(data)