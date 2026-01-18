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
class HorizontalFlipAug(Augmenter):
    """Random horizontal flip.

    Parameters
    ----------
    p : float
        Probability to flip image horizontally
    """

    def __init__(self, p):
        super(HorizontalFlipAug, self).__init__(p=p)
        self.p = p

    def __call__(self, src):
        """Augmenter body"""
        if random.random() < self.p:
            src = nd.flip(src, axis=1)
        return src