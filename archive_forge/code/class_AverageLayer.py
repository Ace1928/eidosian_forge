from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class AverageLayer(Layer):
    """
    Average layer.

    Parameters
    ----------
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one.

    """

    def __init__(self, axis=None, dtype=None, keepdims=False):
        self.axis = axis
        self.dtype = dtype
        self.keepdims = keepdims

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
            Averaged data.

        """
        return np.mean(data, axis=self.axis, dtype=self.dtype, keepdims=self.keepdims)