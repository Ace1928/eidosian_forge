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
class RandomOrderAug(Augmenter):
    """Apply list of augmenters in random order

    Parameters
    ----------
    ts : list of augmenters
        A series of augmenters to be applied in random order
    """

    def __init__(self, ts):
        super(RandomOrderAug, self).__init__()
        self.ts = ts

    def dumps(self):
        """Override the default to avoid duplicate dump."""
        return [self.__class__.__name__.lower(), [x.dumps() for x in self.ts]]

    def __call__(self, src):
        """Augmenter body"""
        random.shuffle(self.ts)
        for t in self.ts:
            src = t(src)
        return src