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
class Matlab2CSVInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='Input MATLAB .mat file')
    reshape_matrix = traits.Bool(True, usedefault=True, desc='The output of this interface is meant for R, so matrices will be        reshaped to vectors by default.')