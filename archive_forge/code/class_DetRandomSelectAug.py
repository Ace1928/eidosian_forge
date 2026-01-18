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
class DetRandomSelectAug(DetAugmenter):
    """Randomly select one augmenter to apply, with chance to skip all.

    Parameters
    ----------
    aug_list : list of DetAugmenter
        The random selection will be applied to one of the augmenters
    skip_prob : float
        The probability to skip all augmenters and return input directly
    """

    def __init__(self, aug_list, skip_prob=0):
        super(DetRandomSelectAug, self).__init__(skip_prob=skip_prob)
        if not isinstance(aug_list, (list, tuple)):
            aug_list = [aug_list]
        for aug in aug_list:
            if not isinstance(aug, DetAugmenter):
                raise ValueError('Allow DetAugmenter in list only')
        if not aug_list:
            skip_prob = 1
        self.aug_list = aug_list
        self.skip_prob = skip_prob

    def dumps(self):
        """Override default."""
        return [self.__class__.__name__.lower(), [x.dumps() for x in self.aug_list]]

    def __call__(self, src, label):
        """Augmenter implementation body"""
        if random.random() < self.skip_prob:
            return (src, label)
        else:
            random.shuffle(self.aug_list)
            return self.aug_list[0](src, label)