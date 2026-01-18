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
class SequentialAug(Augmenter):
    """Composing a sequential augmenter list.

    Parameters
    ----------
    ts : list of augmenters
        A series of augmenters to be applied in sequential order.
    """

    def __init__(self, ts):
        super(SequentialAug, self).__init__()
        self.ts = ts

    def dumps(self):
        """Override the default to avoid duplicate dump."""
        return [self.__class__.__name__.lower(), [x.dumps() for x in self.ts]]

    def __call__(self, src):
        """Augmenter body"""
        for aug in self.ts:
            src = aug(src)
        return src