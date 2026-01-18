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
class PARRECImage(SpatialImage):
    """PAR/REC image"""
    header_class = PARRECHeader
    header: PARRECHeader
    valid_exts = ('.rec', '.par')
    files_types = (('image', '.rec'), ('header', '.par'))
    makeable = False
    rw = False
    ImageArrayProxy = PARRECArrayProxy

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, permit_truncated=False, scaling='dv', strict_sort=False):
        """Create PARREC image from file map `file_map`

        Parameters
        ----------
        file_map : dict
            dict with keys ``image, header`` and values being fileholder
            objects for the respective REC and PAR files.
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        permit_truncated : {False, True}, optional, keyword-only
            If False, raise an error for an image where the header shows signs
            that fewer slices / volumes were recorded than were expected.
        scaling : {'dv', 'fp'}, optional, keyword-only
            Scaling method to apply to data (see
            :meth:`PARRECHeader.get_data_scaling`).
        strict_sort : bool, optional, keyword-only
            If True, a larger number of header fields are used while sorting
            the REC data array.  This may produce a different sort order than
            `strict_sort=False`, where volumes are sorted by the order in which
            the slices appear in the .PAR file.
        """
        with file_map['header'].get_prepare_fileobj('rt') as hdr_fobj:
            hdr = klass.header_class.from_fileobj(hdr_fobj, permit_truncated=permit_truncated, strict_sort=strict_sort)
        rec_fobj = file_map['image'].get_prepare_fileobj()
        data = klass.ImageArrayProxy(rec_fobj, hdr, mmap=mmap, scaling=scaling)
        return klass(data, hdr.get_affine(), header=hdr, extra=None, file_map=file_map)

    @classmethod
    def from_filename(klass, filename, *, mmap=True, permit_truncated=False, scaling='dv', strict_sort=False):
        """Create PARREC image from filename `filename`

        Parameters
        ----------
        filename : str
            Filename of "PAR" or "REC" file
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        permit_truncated : {False, True}, optional, keyword-only
            If False, raise an error for an image where the header shows signs
            that fewer slices / volumes were recorded than were expected.
        scaling : {'dv', 'fp'}, optional, keyword-only
            Scaling method to apply to data (see
            :meth:`PARRECHeader.get_data_scaling`).
        strict_sort : bool, optional, keyword-only
            If True, a larger number of header fields are used while sorting
            the REC data array.  This may produce a different sort order than
            `strict_sort=False`, where volumes are sorted by the order in which
            the slices appear in the .PAR file.
        """
        file_map = klass.filespec_to_file_map(filename)
        return klass.from_file_map(file_map, mmap=mmap, permit_truncated=permit_truncated, scaling=scaling, strict_sort=strict_sort)
    load = from_filename