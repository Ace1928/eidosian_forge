from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class StrideLayer(Layer):
    """
    Stride network layer.

    Parameters
    ----------
    block_size : int
        Re-arrange (stride) the data in blocks of given size.

    """

    def __init__(self, block_size):
        self.block_size = block_size

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
            Strided data.

        """
        from ...utils import segment_axis
        data = segment_axis(data, self.block_size, 1, axis=0, end='cut')
        return data.reshape(len(data), -1)