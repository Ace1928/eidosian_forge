import os
import time
import sys
import zlib
from io import BytesIO
import warnings
import numpy as np
import scipy.sparse
from ._byteordercodes import native_code, swapped_code
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio5_utils import VarReader5
from ._mio5_params import (MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES,
from ._streams import ZlibInputStream
class VarWriter5:
    """ Generic matlab matrix writing class """
    mat_tag = np.zeros((), NDT_TAG_FULL)
    mat_tag['mdtype'] = miMATRIX

    def __init__(self, file_writer):
        self.file_stream = file_writer.file_stream
        self.unicode_strings = file_writer.unicode_strings
        self.long_field_names = file_writer.long_field_names
        self.oned_as = file_writer.oned_as
        self._var_name = None
        self._var_is_global = False

    def write_bytes(self, arr):
        self.file_stream.write(arr.tobytes(order='F'))

    def write_string(self, s):
        self.file_stream.write(s)

    def write_element(self, arr, mdtype=None):
        """ write tag and data """
        if mdtype is None:
            mdtype = NP_TO_MTYPES[arr.dtype.str[1:]]
        if arr.dtype.byteorder == swapped_code:
            arr = arr.byteswap().view(arr.dtype.newbyteorder())
        byte_count = arr.size * arr.itemsize
        if byte_count <= 4:
            self.write_smalldata_element(arr, mdtype, byte_count)
        else:
            self.write_regular_element(arr, mdtype, byte_count)

    def write_smalldata_element(self, arr, mdtype, byte_count):
        tag = np.zeros((), NDT_TAG_SMALL)
        tag['byte_count_mdtype'] = (byte_count << 16) + mdtype
        tag['data'] = arr.tobytes(order='F')
        self.write_bytes(tag)

    def write_regular_element(self, arr, mdtype, byte_count):
        tag = np.zeros((), NDT_TAG_FULL)
        tag['mdtype'] = mdtype
        tag['byte_count'] = byte_count
        self.write_bytes(tag)
        self.write_bytes(arr)
        bc_mod_8 = byte_count % 8
        if bc_mod_8:
            self.file_stream.write(b'\x00' * (8 - bc_mod_8))

    def write_header(self, shape, mclass, is_complex=False, is_logical=False, nzmax=0):
        """ Write header for given data options
        shape : sequence
           array shape
        mclass      - mat5 matrix class
        is_complex  - True if matrix is complex
        is_logical  - True if matrix is logical
        nzmax        - max non zero elements for sparse arrays

        We get the name and the global flag from the object, and reset
        them to defaults after we've used them
        """
        name = self._var_name
        is_global = self._var_is_global
        self._mat_tag_pos = self.file_stream.tell()
        self.write_bytes(self.mat_tag)
        af = np.zeros((), NDT_ARRAY_FLAGS)
        af['data_type'] = miUINT32
        af['byte_count'] = 8
        flags = is_complex << 3 | is_global << 2 | is_logical << 1
        af['flags_class'] = mclass | flags << 8
        af['nzmax'] = nzmax
        self.write_bytes(af)
        self.write_element(np.array(shape, dtype='i4'))
        name = np.asarray(name)
        if name == '':
            self.write_smalldata_element(name, miINT8, 0)
        else:
            self.write_element(name, miINT8)
        self._var_name = ''
        self._var_is_global = False

    def update_matrix_tag(self, start_pos):
        curr_pos = self.file_stream.tell()
        self.file_stream.seek(start_pos)
        byte_count = curr_pos - start_pos - 8
        if byte_count >= 2 ** 32:
            raise MatWriteError('Matrix too large to save with Matlab 5 format')
        self.mat_tag['byte_count'] = byte_count
        self.write_bytes(self.mat_tag)
        self.file_stream.seek(curr_pos)

    def write_top(self, arr, name, is_global):
        """ Write variable at top level of mat file

        Parameters
        ----------
        arr : array_like
            array-like object to create writer for
        name : str, optional
            name as it will appear in matlab workspace
            default is empty string
        is_global : {False, True}, optional
            whether variable will be global on load into matlab
        """
        self._var_is_global = is_global
        self._var_name = name
        self.write(arr)

    def write(self, arr):
        """ Write `arr` to stream at top and sub levels

        Parameters
        ----------
        arr : array_like
            array-like object to create writer for
        """
        mat_tag_pos = self.file_stream.tell()
        if scipy.sparse.issparse(arr):
            self.write_sparse(arr)
            self.update_matrix_tag(mat_tag_pos)
            return
        narr = to_writeable(arr)
        if narr is None:
            raise TypeError(f'Could not convert {arr} (type {type(arr)}) to array')
        if isinstance(narr, MatlabObject):
            self.write_object(narr)
        elif isinstance(narr, MatlabFunction):
            raise MatWriteError('Cannot write matlab functions')
        elif narr is EmptyStructMarker:
            self.write_empty_struct()
        elif narr.dtype.fields:
            self.write_struct(narr)
        elif narr.dtype.hasobject:
            self.write_cells(narr)
        elif narr.dtype.kind in ('U', 'S'):
            if self.unicode_strings:
                codec = 'UTF8'
            else:
                codec = 'ascii'
            self.write_char(narr, codec)
        else:
            self.write_numeric(narr)
        self.update_matrix_tag(mat_tag_pos)

    def write_numeric(self, arr):
        imagf = arr.dtype.kind == 'c'
        logif = arr.dtype.kind == 'b'
        try:
            mclass = NP_TO_MXTYPES[arr.dtype.str[1:]]
        except KeyError:
            if imagf:
                arr = arr.astype('c128')
            elif logif:
                arr = arr.astype('i1')
            else:
                arr = arr.astype('f8')
            mclass = mxDOUBLE_CLASS
        self.write_header(matdims(arr, self.oned_as), mclass, is_complex=imagf, is_logical=logif)
        if imagf:
            self.write_element(arr.real)
            self.write_element(arr.imag)
        else:
            self.write_element(arr)

    def write_char(self, arr, codec='ascii'):
        """ Write string array `arr` with given `codec`
        """
        if arr.size == 0 or np.all(arr == ''):
            shape = (0,) * np.max([arr.ndim, 2])
            self.write_header(shape, mxCHAR_CLASS)
            self.write_smalldata_element(arr, miUTF8, 0)
            return
        arr = arr_to_chars(arr)
        shape = arr.shape
        self.write_header(shape, mxCHAR_CLASS)
        if arr.dtype.kind == 'U' and arr.size:
            n_chars = np.prod(shape)
            st_arr = np.ndarray(shape=(), dtype=arr_dtype_number(arr, n_chars), buffer=arr.T.copy())
            st = st_arr.item().encode(codec)
            arr = np.ndarray(shape=(len(st),), dtype='S1', buffer=st)
        self.write_element(arr, mdtype=miUTF8)

    def write_sparse(self, arr):
        """ Sparse matrices are 2D
        """
        A = arr.tocsc()
        A.sort_indices()
        is_complex = A.dtype.kind == 'c'
        is_logical = A.dtype.kind == 'b'
        nz = A.nnz
        self.write_header(matdims(arr, self.oned_as), mxSPARSE_CLASS, is_complex=is_complex, is_logical=is_logical, nzmax=1 if nz == 0 else nz)
        self.write_element(A.indices.astype('i4'))
        self.write_element(A.indptr.astype('i4'))
        self.write_element(A.data.real)
        if is_complex:
            self.write_element(A.data.imag)

    def write_cells(self, arr):
        self.write_header(matdims(arr, self.oned_as), mxCELL_CLASS)
        A = np.atleast_2d(arr).flatten('F')
        for el in A:
            self.write(el)

    def write_empty_struct(self):
        self.write_header((1, 1), mxSTRUCT_CLASS)
        self.write_element(np.array(1, dtype=np.int32))
        self.write_element(np.array([], dtype=np.int8))

    def write_struct(self, arr):
        self.write_header(matdims(arr, self.oned_as), mxSTRUCT_CLASS)
        self._write_items(arr)

    def _write_items(self, arr):
        fieldnames = [f[0] for f in arr.dtype.descr]
        length = max([len(fieldname) for fieldname in fieldnames]) + 1
        max_length = self.long_field_names and 64 or 32
        if length > max_length:
            raise ValueError('Field names are restricted to %d characters' % (max_length - 1))
        self.write_element(np.array([length], dtype='i4'))
        self.write_element(np.array(fieldnames, dtype='S%d' % length), mdtype=miINT8)
        A = np.atleast_2d(arr).flatten('F')
        for el in A:
            for f in fieldnames:
                self.write(el[f])

    def write_object(self, arr):
        """Same as writing structs, except different mx class, and extra
        classname element after header
        """
        self.write_header(matdims(arr, self.oned_as), mxOBJECT_CLASS)
        self.write_element(np.array(arr.classname, dtype='S'), mdtype=miINT8)
        self._write_items(arr)