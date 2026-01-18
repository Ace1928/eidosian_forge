import glob
from os.path import join as pjoin
import numpy as np
from .. import Nifti1Image
from .dicomwrappers import wrapper_from_data, wrapper_from_file
def _slice_sorter(s):
    return s.slice_indicator