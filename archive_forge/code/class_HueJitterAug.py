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
class HueJitterAug(Augmenter):
    """Random hue jitter augmentation.

    Parameters
    ----------
    hue : float
        The hue jitter ratio range, [0, 1]
    """

    def __init__(self, hue):
        super(HueJitterAug, self).__init__(hue=hue)
        self.hue = hue
        self.tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]])
        self.ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.107, 1.705]])

    def __call__(self, src):
        """Augmenter body.
        Using approximate linear transfomation described in:
        https://beesbuzz.biz/code/hsv_color_transforms.php
        """
        alpha = random.uniform(-self.hue, self.hue)
        u = np.cos(alpha * np.pi)
        w = np.sin(alpha * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        t = np.dot(np.dot(self.ityiq, bt), self.tyiq).T
        src = nd.dot(src, nd.array(t))
        return src