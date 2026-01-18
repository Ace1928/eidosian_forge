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
class ResizeAug(Augmenter):
    """Make resize shorter edge to size augmenter.

    Parameters
    ----------
    size : int
        The length to be set for the shorter edge.
    interp : int, optional, default=2
        Interpolation method. See resize_short for details.
    """

    def __init__(self, size, interp=2):
        super(ResizeAug, self).__init__(size=size, interp=interp)
        self.size = size
        self.interp = interp

    def __call__(self, src):
        """Augmenter body"""
        return resize_short(src, self.size, self.interp)