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
def _chk_trunc(idef_name, gdef_max_name):
    if gdef_max_name not in general_info:
        return
    id_values = image_defs[idef_name + ' number']
    n_have = len(set(id_values))
    n_expected = general_info[gdef_max_name]
    if n_have != n_expected:
        _err_or_warn(f'Header inconsistency: Found {n_have} {idef_name} values, but expected {n_expected}')