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
def _data_from_rec(rec_fileobj, in_shape, dtype, slice_indices, out_shape, scalings=None, mmap=True):
    """Load and return array data from REC file

    Parameters
    ----------
    rec_fileobj : file-like
        The file to process.
    in_shape : tuple
        The input shape inferred from the PAR file.
    dtype : dtype
        The datatype.
    slice_indices : array of int
        The indices used to re-index the resulting array properly.
    out_shape : tuple
        The output shape.
    scalings : {None, sequence}, optional
        Scalings to use. If not None, a length 2 sequence giving (``slope``,
        ``intercept``), where ``slope`` and ``intercept`` are arrays that can
        be broadcast to `out_shape`.
    mmap : {True, False, 'c', 'r', 'r+'}, optional
        `mmap` controls the use of numpy memory mapping for reading data.  If
        False, do not try numpy ``memmap`` for data array.  If one of {'c',
        'r', 'r+'}, try numpy memmap with ``mode=mmap``.  A `mmap` value of
        True gives the same behavior as ``mmap='c'``.  If `rec_fileobj` cannot
        be memory-mapped, ignore `mmap` value and read array from file.

    Returns
    -------
    data : array
        The scaled and sorted array.
    """
    rec_data = array_from_file(in_shape, dtype, rec_fileobj, mmap=mmap)
    rec_data = rec_data[..., slice_indices]
    rec_data = rec_data.reshape(out_shape, order='F')
    if scalings is not None:
        rec_data = rec_data * scalings[0]
        rec_data += scalings[1]
    return rec_data