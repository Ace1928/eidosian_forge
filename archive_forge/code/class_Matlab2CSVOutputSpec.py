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
class Matlab2CSVOutputSpec(TraitedSpec):
    csv_files = OutputMultiPath(File(desc='Output CSV files for each variable saved in the input .mat        file'))