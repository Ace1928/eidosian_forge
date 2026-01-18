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
def _matlab2csv(in_array, name, reshape):
    output_array = np.asarray(in_array)
    if reshape:
        if len(np.shape(output_array)) > 1:
            output_array = np.reshape(output_array, (np.shape(output_array)[0] * np.shape(output_array)[1], 1))
            iflogger.info(np.shape(output_array))
    output_name = op.abspath(name + '.csv')
    np.savetxt(output_name, output_array, delimiter=',')
    return output_name