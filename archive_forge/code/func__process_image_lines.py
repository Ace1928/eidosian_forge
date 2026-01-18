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
def _process_image_lines(image_lines, version):
    """Process image information definition lines according to `version`"""
    image_def_dtd = image_def_dtds[version]
    image_defs = np.zeros(len(image_lines), dtype=image_def_dtd)
    for i, line in enumerate(image_lines):
        items = line.split()
        item_counter = 0
        for props in image_def_dtd:
            if len(props) == 2:
                name, np_type = props
                value = items[item_counter]
                if not np.dtype(np_type).kind == 'S':
                    value = np_type(value)
                item_counter += 1
            elif len(props) == 3:
                name, np_type, shape = props
                nelements = np.prod(shape)
                value = items[item_counter:item_counter + nelements]
                value = [np_type(v) for v in value]
                item_counter += nelements
            image_defs[name][i] = value
    return image_defs