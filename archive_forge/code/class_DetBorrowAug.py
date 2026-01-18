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
class DetBorrowAug(DetAugmenter):
    """Borrow standard augmenter from image classification.
    Which is good once you know label won't be affected after this augmenter.

    Parameters
    ----------
    augmenter : mx.image.Augmenter
        The borrowed standard augmenter which has no effect on label
    """

    def __init__(self, augmenter):
        if not isinstance(augmenter, Augmenter):
            raise TypeError('Borrowing from invalid Augmenter')
        super(DetBorrowAug, self).__init__(augmenter=augmenter.dumps())
        self.augmenter = augmenter

    def dumps(self):
        """Override the default one to avoid duplicate dump."""
        return [self.__class__.__name__.lower(), self.augmenter.dumps()]

    def __call__(self, src, label):
        """Augmenter implementation body"""
        src = self.augmenter(src)
        return (src, label)