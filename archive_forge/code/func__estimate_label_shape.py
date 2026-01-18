import json
import logging
import random
import warnings
import numpy as np
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray._internal import _cvcopyMakeBorder as copyMakeBorder
from .. import io
from .image import RandomOrderAug, ColorJitterAug, LightingAug, ColorNormalizeAug
from .image import ResizeAug, ForceResizeAug, CastAug, HueJitterAug, RandomGrayAug
from .image import fixed_crop, ImageIter, Augmenter
from ..util import is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def _estimate_label_shape(self):
    """Helper function to estimate label shape"""
    max_count = 0
    self.reset()
    try:
        while True:
            label, _ = self.next_sample()
            label = self._parse_label(label)
            max_count = max(max_count, label.shape[0])
    except StopIteration:
        pass
    self.reset()
    return (max_count, label.shape[1])