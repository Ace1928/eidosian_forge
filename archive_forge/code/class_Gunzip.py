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
class Gunzip(Gzip):
    """Gunzip wrapper

    >>> from nipype.algorithms.misc import Gunzip
    >>> gunzip = Gunzip(in_file='tpms_msk.nii.gz')
    >>> res = gunzip.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../tpms_msk.nii'

    .. testcleanup::

    >>> os.unlink('tpms_msk.nii')
    """
    input_spec = GunzipInputSpec