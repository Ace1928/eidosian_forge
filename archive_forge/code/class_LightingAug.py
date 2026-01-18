import sys
import os
import random
import logging
import json
import warnings
from numbers import Number
import numpy as np
from .. import numpy as _mx_np  # pylint: disable=reimported
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray import _internal
from .. import io
from .. import recordio
from .. util import is_np_array
from ..ndarray.numpy import _internal as _npi
class LightingAug(Augmenter):
    """Add PCA based noise.

    Parameters
    ----------
    alphastd : float
        Noise level
    eigval : 3x1 np.array
        Eigen values
    eigvec : 3x3 np.array
        Eigen vectors
    """

    def __init__(self, alphastd, eigval, eigvec):
        super(LightingAug, self).__init__(alphastd=alphastd, eigval=eigval, eigvec=eigvec)
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, src):
        """Augmenter body"""
        alpha = np.random.normal(0, self.alphastd, size=(3,))
        rgb = np.dot(self.eigvec * alpha, self.eigval)
        src += nd.array(rgb)
        return src