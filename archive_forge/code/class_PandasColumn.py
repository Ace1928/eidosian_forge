from __future__ import annotations
from typing import Any
import numpy as np
from pandas._libs.lib import infer_dtype
from pandas._libs.tslibs import iNaT
from pandas.errors import NoBufferPresent
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.core.interchange.buffer import PandasBuffer
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.utils import (
class PandasColumn(Column):
    """
    A column object, with only the methods and properties required by the
    interchange protocol defined.
    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length
    strings).
    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.
    """

    def __init__(self, column: pd.Series, allow_copy: bool=True) -> None:
        """
        Note: doesn't deal with extension arrays yet, just assume a regular
        Series/ndarray for now.
        """
        if isinstance(column, pd.DataFrame):
            raise TypeError(f'Expected a Series, got a DataFrame. This likely happened because you called __dataframe__ on a DataFrame which, after converting column names to string, resulted in duplicated names: {column.columns}. Please rename these columns before using the interchange protocol.')
        if not isinstance(column, pd.Series):
            raise NotImplementedError(f'Columns of type {type(column)} not handled yet')
        self._col = column
        self._allow_copy = allow_copy

    def size(self) -> int:
        """
        Size of the column, in elements.
        """
        return self._col.size

    @property
    def offset(self) -> int:
        """
        Offset of first element. Always zero.
        """
        return 0

    @cache_readonly
    def dtype(self) -> tuple[DtypeKind, int, str, str]:
        dtype = self._col.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            codes = self._col.values.codes
            _, bitwidth, c_arrow_dtype_f_str, _ = self._dtype_from_pandasdtype(codes.dtype)
            return (DtypeKind.CATEGORICAL, bitwidth, c_arrow_dtype_f_str, Endianness.NATIVE)
        elif is_string_dtype(dtype):
            if infer_dtype(self._col) in ('string', 'empty'):
                return (DtypeKind.STRING, 8, dtype_to_arrow_c_fmt(dtype), Endianness.NATIVE)
            raise NotImplementedError('Non-string object dtypes are not supported yet')
        else:
            return self._dtype_from_pandasdtype(dtype)

    def _dtype_from_pandasdtype(self, dtype) -> tuple[DtypeKind, int, str, str]:
        """
        See `self.dtype` for details.
        """
        kind = _NP_KINDS.get(dtype.kind, None)
        if kind is None:
            raise ValueError(f'Data type {dtype} not supported by interchange protocol')
        if isinstance(dtype, ArrowDtype):
            byteorder = dtype.numpy_dtype.byteorder
        elif isinstance(dtype, DatetimeTZDtype):
            byteorder = dtype.base.byteorder
        elif isinstance(dtype, BaseMaskedDtype):
            byteorder = dtype.numpy_dtype.byteorder
        else:
            byteorder = dtype.byteorder
        return (kind, dtype.itemsize * 8, dtype_to_arrow_c_fmt(dtype), byteorder)

    @property
    def describe_categorical(self):
        """
        If the dtype is categorical, there are two options:
        - There are only values in the data buffer.
        - There is a separate non-categorical Column encoding for categorical values.

        Raises TypeError if the dtype is not categorical

        Content of returned dict:
            - "is_ordered" : bool, whether the ordering of dictionary indices is
                             semantically meaningful.
            - "is_dictionary" : bool, whether a dictionary-style mapping of
                                categorical values to other objects exists
            - "categories" : Column representing the (implicit) mapping of indices to
                             category values (e.g. an array of cat1, cat2, ...).
                             None if not a dictionary-style categorical.
        """
        if not self.dtype[0] == DtypeKind.CATEGORICAL:
            raise TypeError('describe_categorical only works on a column with categorical dtype!')
        return {'is_ordered': self._col.cat.ordered, 'is_dictionary': True, 'categories': PandasColumn(pd.Series(self._col.cat.categories))}

    @property
    def describe_null(self):
        kind = self.dtype[0]
        try:
            null, value = _NULL_DESCRIPTION[kind]
        except KeyError:
            raise NotImplementedError(f'Data type {kind} not yet supported')
        return (null, value)

    @cache_readonly
    def null_count(self) -> int:
        """
        Number of null elements. Should always be known.
        """
        return self._col.isna().sum().item()

    @property
    def metadata(self) -> dict[str, pd.Index]:
        """
        Store specific metadata of the column.
        """
        return {'pandas.index': self._col.index}

    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.
        """
        return 1

    def get_chunks(self, n_chunks: int | None=None):
        """
        Return an iterator yielding the chunks.
        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
        if n_chunks and n_chunks > 1:
            size = len(self._col)
            step = size // n_chunks
            if size % n_chunks != 0:
                step += 1
            for start in range(0, step * n_chunks, step):
                yield PandasColumn(self._col.iloc[start:start + step], self._allow_copy)
        else:
            yield self

    def get_buffers(self) -> ColumnBuffers:
        """
        Return a dictionary containing the underlying buffers.
        The returned dictionary has the following contents:
            - "data": a two-element tuple whose first element is a buffer
                      containing the data and whose second element is the data
                      buffer's associated dtype.
            - "validity": a two-element tuple whose first element is a buffer
                          containing mask values indicating missing data and
                          whose second element is the mask value buffer's
                          associated dtype. None if the null representation is
                          not a bit or byte mask.
            - "offsets": a two-element tuple whose first element is a buffer
                         containing the offset values for variable-size binary
                         data (e.g., variable-length strings) and whose second
                         element is the offsets buffer's associated dtype. None
                         if the data buffer does not have an associated offsets
                         buffer.
        """
        buffers: ColumnBuffers = {'data': self._get_data_buffer(), 'validity': None, 'offsets': None}
        try:
            buffers['validity'] = self._get_validity_buffer()
        except NoBufferPresent:
            pass
        try:
            buffers['offsets'] = self._get_offsets_buffer()
        except NoBufferPresent:
            pass
        return buffers

    def _get_data_buffer(self) -> tuple[PandasBuffer, Any]:
        """
        Return the buffer containing the data and the buffer's associated dtype.
        """
        if self.dtype[0] in (DtypeKind.INT, DtypeKind.UINT, DtypeKind.FLOAT, DtypeKind.BOOL, DtypeKind.DATETIME):
            if self.dtype[0] == DtypeKind.DATETIME and len(self.dtype[2]) > 4:
                np_arr = self._col.dt.tz_convert(None).to_numpy()
            else:
                np_arr = self._col.to_numpy()
            buffer = PandasBuffer(np_arr, allow_copy=self._allow_copy)
            dtype = self.dtype
        elif self.dtype[0] == DtypeKind.CATEGORICAL:
            codes = self._col.values._codes
            buffer = PandasBuffer(codes, allow_copy=self._allow_copy)
            dtype = self._dtype_from_pandasdtype(codes.dtype)
        elif self.dtype[0] == DtypeKind.STRING:
            buf = self._col.to_numpy()
            b = bytearray()
            for obj in buf:
                if isinstance(obj, str):
                    b.extend(obj.encode(encoding='utf-8'))
            buffer = PandasBuffer(np.frombuffer(b, dtype='uint8'))
            dtype = self.dtype
        else:
            raise NotImplementedError(f'Data type {self._col.dtype} not handled yet')
        return (buffer, dtype)

    def _get_validity_buffer(self) -> tuple[PandasBuffer, Any]:
        """
        Return the buffer containing the mask values indicating missing data and
        the buffer's associated dtype.
        Raises NoBufferPresent if null representation is not a bit or byte mask.
        """
        null, invalid = self.describe_null
        if self.dtype[0] == DtypeKind.STRING:
            buf = self._col.to_numpy()
            valid = invalid == 0
            invalid = not valid
            mask = np.zeros(shape=(len(buf),), dtype=np.bool_)
            for i, obj in enumerate(buf):
                mask[i] = valid if isinstance(obj, str) else invalid
            buffer = PandasBuffer(mask)
            dtype = (DtypeKind.BOOL, 8, ArrowCTypes.BOOL, Endianness.NATIVE)
            return (buffer, dtype)
        try:
            msg = f'{_NO_VALIDITY_BUFFER[null]} so does not have a separate mask'
        except KeyError:
            raise NotImplementedError('See self.describe_null')
        raise NoBufferPresent(msg)

    def _get_offsets_buffer(self) -> tuple[PandasBuffer, Any]:
        """
        Return the buffer containing the offset values for variable-size binary
        data (e.g., variable-length strings) and the buffer's associated dtype.
        Raises NoBufferPresent if the data buffer does not have an associated
        offsets buffer.
        """
        if self.dtype[0] == DtypeKind.STRING:
            values = self._col.to_numpy()
            ptr = 0
            offsets = np.zeros(shape=(len(values) + 1,), dtype=np.int64)
            for i, v in enumerate(values):
                if isinstance(v, str):
                    b = v.encode(encoding='utf-8')
                    ptr += len(b)
                offsets[i + 1] = ptr
            buffer = PandasBuffer(offsets)
            dtype = (DtypeKind.INT, 64, ArrowCTypes.INT64, Endianness.NATIVE)
        else:
            raise NoBufferPresent('This column has a fixed-length dtype so it does not have an offsets buffer')
        return (buffer, dtype)