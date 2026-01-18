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
class MergeROIsInputSpec(TraitedSpec):
    in_files = InputMultiPath(File(exists=True, mandatory=True, desc='files to be re-merged'))
    in_index = InputMultiPath(File(exists=True, mandatory=True), desc='array keeping original locations')
    in_reference = File(exists=True, desc='reference file')