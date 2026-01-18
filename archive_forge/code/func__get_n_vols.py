import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from locale import getpreferredencoding
import numpy as np
from .affines import apply_affine, dot_reduce, from_matvec
from .eulerangles import euler2mat
from .fileslice import fileslice, strided_scalar
from .nifti1 import unit_codes
from .openers import ImageOpener
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import Recoder, array_from_file
def _get_n_vols(self):
    """Get number of volumes for output data"""
    slice_nos = self.image_defs['slice number']
    vol_nos = vol_numbers(slice_nos)
    is_full = vol_is_full(slice_nos, self.general_info['max_slices'])
    return len(set(np.array(vol_nos)[is_full]))