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
class ColorNormalizeAug(Augmenter):
    """Mean and std normalization.

    Parameters
    ----------
    mean : NDArray
        RGB mean to be subtracted
    std : NDArray
        RGB standard deviation to be divided
    """

    def __init__(self, mean, std):
        super(ColorNormalizeAug, self).__init__(mean=mean, std=std)
        self.mean = mean if mean is None or isinstance(mean, nd.NDArray) else nd.array(mean)
        self.std = std if std is None or isinstance(std, nd.NDArray) else nd.array(std)

    def __call__(self, src):
        """Augmenter body"""
        return color_normalize(src, self.mean, self.std)