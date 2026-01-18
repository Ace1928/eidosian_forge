from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class BidirectionalLayer(Layer):
    """
    Bidirectional network layer.

    Parameters
    ----------
    fwd_layer : Layer instance
        Forward layer.
    bwd_layer : Layer instance
        Backward layer.

    """

    def __init__(self, fwd_layer, bwd_layer):
        self.fwd_layer = fwd_layer
        self.bwd_layer = bwd_layer

    def activate(self, data, **kwargs):
        """
        Activate the layer.

        After activating the `fwd_layer` with the data and the `bwd_layer` with
        the data in reverse temporal order, the two activations are stacked and
        returned.

        Parameters
        ----------
        data : numpy array, shape (num_frames, num_inputs)
            Activate with this data.

        Returns
        -------
        numpy array, shape (num_frames, num_hiddens)
            Activations for this data.

        """
        fwd = self.fwd_layer(data, **kwargs)
        bwd = self.bwd_layer(data[::-1], **kwargs)
        return np.hstack((fwd, bwd[::-1]))