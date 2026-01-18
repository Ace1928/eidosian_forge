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
def _get_unscaled(self, slicer):
    indices = self._slice_indices
    if slicer == ():
        with ImageOpener(self.file_like) as fileobj:
            rec_data = array_from_file(self._rec_shape, self._dtype, fileobj, mmap=self._mmap)
            rec_data = rec_data[..., indices]
            return rec_data.reshape(self._shape, order='F')
    elif indices[0] != 0 or np.any(np.diff(indices) != 1):
        return self._get_unscaled(())[slicer]
    with ImageOpener(self.file_like) as fileobj:
        return fileslice(fileobj, slicer, self._shape, self._dtype, 0, 'F')