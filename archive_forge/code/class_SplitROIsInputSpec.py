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
class SplitROIsInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='file to be split')
    in_mask = File(exists=True, desc='only process files inside mask')
    roi_size = traits.Tuple(traits.Int, traits.Int, traits.Int, desc='desired ROI size')