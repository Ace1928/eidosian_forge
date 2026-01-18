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
def _check_satisfy_constraints(self, label, xmin, ymin, xmax, ymax, width, height):
    """Check if constrains are satisfied"""
    if (xmax - xmin) * (ymax - ymin) < 2:
        return False
    x1 = float(xmin) / width
    y1 = float(ymin) / height
    x2 = float(xmax) / width
    y2 = float(ymax) / height
    object_areas = self._calculate_areas(label[:, 1:])
    valid_objects = np.where(object_areas * width * height > 2)[0]
    if valid_objects.size < 1:
        return False
    intersects = self._intersect(label[valid_objects, 1:], x1, y1, x2, y2)
    coverages = self._calculate_areas(intersects) / object_areas[valid_objects]
    coverages = coverages[np.where(coverages > 0)[0]]
    return coverages.size > 0 and np.amin(coverages) > self.min_object_covered