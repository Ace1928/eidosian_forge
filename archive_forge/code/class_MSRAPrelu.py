import re
import logging
import warnings
import json
from math import sqrt
import numpy as np
from .base import string_types
from .ndarray import NDArray, load
from . import random
from . import registry
from . import ndarray
from . util import is_np_array
from . import numpy as _mx_np  # pylint: disable=reimported
@register
class MSRAPrelu(Xavier):
    """Initialize the weight according to a MSRA paper.

    This initializer implements *Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification*, available at
    https://arxiv.org/abs/1502.01852.

    This initializer is proposed for initialization related to ReLu activation,
    it makes some changes on top of Xavier method.

    Parameters
    ----------
    factor_type: str, optional
        Can be ``'avg'``, ``'in'``, or ``'out'``.

    slope: float, optional
        initial slope of any PReLU (or similar) nonlinearities.
    """

    def __init__(self, factor_type='avg', slope=0.25):
        magnitude = 2.0 / (1 + slope ** 2)
        super(MSRAPrelu, self).__init__('gaussian', factor_type, magnitude)
        self._kwargs = {'factor_type': factor_type, 'slope': slope}