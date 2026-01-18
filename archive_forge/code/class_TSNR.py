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
class TSNR(confounds.TSNR):
    """
    .. deprecated:: 0.12.1
       Use :py:class:`nipype.algorithms.confounds.TSNR` instead
    """

    def __init__(self, **inputs):
        super(confounds.TSNR, self).__init__(**inputs)
        warnings.warn('This interface has been moved since 0.12.0, please use nipype.algorithms.confounds.TSNR', UserWarning)