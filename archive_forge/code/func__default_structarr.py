import os
import string
import struct
import warnings
import numpy as np
import nibabel as nib
from nibabel.openers import Opener
from nibabel.orientations import aff2axcodes, axcodes2ornt
from nibabel.volumeutils import endian_codes, native_code, swapped_code
from .array_sequence import create_arraysequences_from_generator
from .header import Field
from .tractogram import LazyTractogram, Tractogram, TractogramItem
from .tractogram_file import DataError, HeaderError, HeaderWarning, TractogramFile
from .utils import peek_next
@classmethod
def _default_structarr(cls, endianness=None):
    """Return an empty compliant TRK header as numpy structured array"""
    dt = header_2_dtype
    if endianness is not None:
        endianness = endian_codes[endianness]
        dt = dt.newbyteorder(endianness)
    st_arr = np.zeros((), dtype=dt)
    st_arr[Field.MAGIC_NUMBER] = cls.MAGIC_NUMBER
    st_arr[Field.VOXEL_SIZES] = np.array((1, 1, 1), dtype='f4')
    st_arr[Field.DIMENSIONS] = np.array((1, 1, 1), dtype='h')
    st_arr[Field.VOXEL_TO_RASMM] = np.eye(4, dtype='f4')
    st_arr[Field.VOXEL_ORDER] = b'RAS'
    st_arr['version'] = 2
    st_arr['hdr_size'] = cls.HEADER_SIZE
    return st_arr