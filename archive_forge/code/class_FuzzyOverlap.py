import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class FuzzyOverlap(nam.FuzzyOverlap):
    """Calculates various overlap measures between two maps, using a fuzzy
    definition.

    .. deprecated:: 0.10.0
       Use :py:class:`nipype.algorithms.metrics.FuzzyOverlap` instead.
    """

    def __init__(self, **inputs):
        super(nam.FuzzyOverlap, self).__init__(**inputs)
        warnings.warn('This interface has been deprecated since 0.10.0, please use nipype.algorithms.metrics.FuzzyOverlap', DeprecationWarning)