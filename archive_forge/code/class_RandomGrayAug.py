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
class RandomGrayAug(Augmenter):
    """Randomly convert to gray image.

    Parameters
    ----------
    p : float
        Probability to convert to grayscale
    """

    def __init__(self, p):
        super(RandomGrayAug, self).__init__(p=p)
        self.p = p
        self.mat = nd.array([[0.21, 0.21, 0.21], [0.72, 0.72, 0.72], [0.07, 0.07, 0.07]])

    def __call__(self, src):
        """Augmenter body"""
        if random.random() < self.p:
            src = nd.dot(src, self.mat)
        return src