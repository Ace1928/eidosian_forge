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
class DetAugmenter(object):
    """Detection base augmenter"""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in self._kwargs.items():
            if isinstance(v, nd.NDArray):
                v = v.asnumpy()
            if isinstance(v, np.ndarray):
                v = v.tolist()
                self._kwargs[k] = v

    def dumps(self):
        """Saves the Augmenter to string

        Returns
        -------
        str
            JSON formatted string that describes the Augmenter.
        """
        return json.dumps([self.__class__.__name__.lower(), self._kwargs])

    def __call__(self, src, label):
        """Abstract implementation body"""
        raise NotImplementedError('Must override implementation.')