from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class BatchNormLayer(Layer):
    """
    Batch normalization layer with activation function. The previous layer
    is usually linear with no bias - the BatchNormLayer's beta parameter
    replaces it. See [1]_ for a detailed understanding of the parameters.

    Parameters
    ----------
    beta : numpy array
        Values for the `beta` parameter. Must be broadcastable to the incoming
        shape.
    gamma : numpy array
        Values for the `gamma` parameter. Must be broadcastable to the incoming
        shape.
    mean : numpy array
        Mean values of incoming data. Must be broadcastable to the incoming
        shape.
    inv_std : numpy array
        Inverse standard deviation of incoming data. Must be broadcastable to
        the incoming shape.
    activation_fn : numpy ufunc
        Activation function.

    References
    ----------
    .. [1] "Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift"
           Sergey Ioffe and Christian Szegedy.
           http://arxiv.org/abs/1502.03167, 2015.

    """

    def __init__(self, beta, gamma, mean, inv_std, activation_fn):
        self.beta = beta
        self.gamma = gamma
        self.mean = mean
        self.inv_std = inv_std
        self.activation_fn = activation_fn

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
            Normalized data.

        """
        return self.activation_fn((data - self.mean) * (self.gamma * self.inv_std) + self.beta)