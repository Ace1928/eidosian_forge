from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class ReshapeLayer(Layer):
    """
    Reshape Layer.

    Parameters
    ----------
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    order : {'C', 'F', 'A'}, optional
        Index order or the input. See np.reshape for a detailed description.

    """

    def __init__(self, newshape, order='C'):
        self.newshape = newshape
        self.order = order

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
            Reshaped data.

        """
        return np.reshape(data, self.newshape, self.order)