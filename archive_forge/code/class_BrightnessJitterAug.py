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
class BrightnessJitterAug(Augmenter):
    """Random brightness jitter augmentation.

    Parameters
    ----------
    brightness : float
        The brightness jitter ratio range, [0, 1]
    """

    def __init__(self, brightness):
        super(BrightnessJitterAug, self).__init__(brightness=brightness)
        self.brightness = brightness

    def __call__(self, src):
        """Augmenter body"""
        alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
        src *= alpha
        return src