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
def _random_crop_proposal(self, label, height, width):
    """Propose cropping areas"""
    from math import sqrt
    if not self.enabled or height <= 0 or width <= 0:
        return ()
    min_area = self.area_range[0] * height * width
    max_area = self.area_range[1] * height * width
    for _ in range(self.max_attempts):
        ratio = random.uniform(*self.aspect_ratio_range)
        if ratio <= 0:
            continue
        h = int(round(sqrt(min_area / ratio)))
        max_h = int(round(sqrt(max_area / ratio)))
        if round(max_h * ratio) > width:
            max_h = int((width + 0.4999999) / ratio)
        if max_h > height:
            max_h = height
        if h > max_h:
            h = max_h
        if h < max_h:
            h = random.randint(h, max_h)
        w = int(round(h * ratio))
        assert w <= width
        area = w * h
        if area < min_area:
            h += 1
            w = int(round(h * ratio))
            area = w * h
        if area > max_area:
            h -= 1
            w = int(round(h * ratio))
            area = w * h
        if not (min_area <= area <= max_area and 0 <= w <= width and (0 <= h <= height)):
            continue
        y = random.randint(0, max(0, height - h))
        x = random.randint(0, max(0, width - w))
        if self._check_satisfy_constraints(label, x, y, x + w, y + h, width, height):
            new_label = self._update_labels(label, (x, y, w, h), height, width)
            if new_label is not None:
                return (x, y, w, h, new_label)
    return ()