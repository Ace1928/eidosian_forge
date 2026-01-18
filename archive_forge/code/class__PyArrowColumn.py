from __future__ import annotations
import enum
from typing import (
import sys
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.interchange.buffer import _PyArrowBuffer
class _PyArrowColumn:
    """
    A column object, with only the methods and properties required by the
    interchange protocol defined.

    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length
    strings).

    TBD: Arrow has a separate "null" dtype, and has no separate mask concept.
         Instead, it seems to use "children" for both columns with a bit mask,
         and for nested dtypes. Unclear whether this is elegant or confusing.
         This design requires checking the null representation explicitly.

         The Arrow design requires checking:
         1. the ARROW_FLAG_NULLABLE (for sentinel values)
         2. if a column has two children, combined with one of those children
            having a null dtype.

         Making the mask concept explicit seems useful. One null dtype would
         not be enough to cover both bit and byte masks, so that would mean
         even more checking if we did it the Arrow way.

    TBD: there's also the "chunk" concept here, which is implicit in Arrow as
         multiple buffers per array (= column here). Semantically it may make
         sense to have both: chunks were meant for example for lazy evaluation
         of data which doesn't fit in memory, while multiple buffers per column
         could also come from doing a selection operation on a single
         contiguous buffer.

         Given these concepts, one would expect chunks to be all of the same
         size (say a 10,000 row dataframe could have 10 chunks of 1,000 rows),
         while multiple buffers could have data-dependent lengths. Not an issue
         in pandas if one column is backed by a single NumPy array, but in
         Arrow it seems possible.
         Are multiple chunks *and* multiple buffers per column necessary for
         the purposes of this interchange protocol, or must producers either
         reuse the chunk concept for this or copy the data?

    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.
    """

    def __init__(self, column: pa.Array | pa.ChunkedArray, allow_copy: bool=True) -> None:
        """
        Handles PyArrow Arrays and ChunkedArrays.
        """
        if isinstance(column, pa.ChunkedArray):
            if column.num_chunks == 1:
                column = column.chunk(0)
            else:
                if not allow_copy:
                    raise RuntimeError('Chunks will be combined and a copy is required which is forbidden by allow_copy=False')
                column = column.combine_chunks()
        self._allow_copy = allow_copy
        if pa.types.is_boolean(column.type):
            if not allow_copy:
                raise RuntimeError('Boolean column will be casted to uint8 and a copy is required which is forbidden by allow_copy=False')
            self._dtype = self._dtype_from_arrowdtype(column.type, 8)
            self._col = pc.cast(column, pa.uint8())
        else:
            self._col = column
            dtype = self._col.type
            try:
                bit_width = dtype.bit_width
            except ValueError:
                bit_width = 8
            self._dtype = self._dtype_from_arrowdtype(dtype, bit_width)

    def size(self) -> int:
        """
        Size of the column, in elements.

        Corresponds to DataFrame.num_rows() if column is a single chunk;
        equal to size of this current chunk otherwise.

        Is a method rather than a property because it may cause a (potentially
        expensive) computation for some dataframe implementations.
        """
        return len(self._col)

    @property
    def offset(self) -> int:
        """
        Offset of first element.

        May be > 0 if using chunks; for example for a column with N chunks of
        equal size M (only the last chunk may be shorter),
        ``offset = n * M``, ``n = 0 .. N-1``.
        """
        return self._col.offset

    @property
    def dtype(self) -> Tuple[DtypeKind, int, str, str]:
        """
        Dtype description as a tuple ``(kind, bit-width, format string,
        endianness)``.

        Bit-width : the number of bits as an integer
        Format string : data type description format string in Apache Arrow C
                        Data Interface format.
        Endianness : current only native endianness (``=``) is supported

        Notes:
            - Kind specifiers are aligned with DLPack where possible (hence the
              jump to 20, leave enough room for future extension)
            - Masks must be specified as boolean with either bit width 1 (for
              bit masks) or 8 (for byte masks).
            - Dtype width in bits was preferred over bytes
            - Endianness isn't too useful, but included now in case in the
              future we need to support non-native endianness
            - Went with Apache Arrow format strings over NumPy format strings
              because they're more complete from a dataframe perspective
            - Format strings are mostly useful for datetime specification, and
              for categoricals.
            - For categoricals, the format string describes the type of the
              categorical in the data buffer. In case of a separate encoding of
              the categorical (e.g. an integer to string mapping), this can
              be derived from ``self.describe_categorical``.
            - Data types not included: complex, Arrow-style null, binary,
              decimal, and nested (list, struct, map, union) dtypes.
        """
        return self._dtype

    def _dtype_from_arrowdtype(self, dtype: pa.DataType, bit_width: int) -> Tuple[DtypeKind, int, str, str]:
        """
        See `self.dtype` for details.
        """
        if pa.types.is_timestamp(dtype):
            kind = DtypeKind.DATETIME
            ts = dtype.unit[0]
            tz = dtype.tz if dtype.tz else ''
            f_string = 'ts{ts}:{tz}'.format(ts=ts, tz=tz)
            return (kind, bit_width, f_string, Endianness.NATIVE)
        elif pa.types.is_dictionary(dtype):
            kind = DtypeKind.CATEGORICAL
            arr = self._col
            indices_dtype = arr.indices.type
            _, f_string = _PYARROW_KINDS.get(indices_dtype)
            return (kind, bit_width, f_string, Endianness.NATIVE)
        else:
            kind, f_string = _PYARROW_KINDS.get(dtype, (None, None))
            if kind is None:
                raise ValueError(f'Data type {dtype} not supported by interchange protocol')
            return (kind, bit_width, f_string, Endianness.NATIVE)

    @property
    def describe_categorical(self) -> CategoricalDescription:
        """
        If the dtype is categorical, there are two options:
        - There are only values in the data buffer.
        - There is a separate non-categorical Column encoding categorical
          values.

        Raises TypeError if the dtype is not categorical

        Returns the dictionary with description on how to interpret the
        data buffer:
            - "is_ordered" : bool, whether the ordering of dictionary indices
                             is semantically meaningful.
            - "is_dictionary" : bool, whether a mapping of
                                categorical values to other objects exists
            - "categories" : Column representing the (implicit) mapping of
                             indices to category values (e.g. an array of
                             cat1, cat2, ...). None if not a dictionary-style
                             categorical.

        TBD: are there any other in-memory representations that are needed?
        """
        arr = self._col
        if not pa.types.is_dictionary(arr.type):
            raise TypeError('describe_categorical only works on a column with categorical dtype!')
        return {'is_ordered': self._col.type.ordered, 'is_dictionary': True, 'categories': _PyArrowColumn(arr.dictionary)}

    @property
    def describe_null(self) -> Tuple[ColumnNullType, Any]:
        """
        Return the missing value (or "null") representation the column dtype
        uses, as a tuple ``(kind, value)``.

        Value : if kind is "sentinel value", the actual value. If kind is a bit
        mask or a byte mask, the value (0 or 1) indicating a missing value.
        None otherwise.
        """
        if self.null_count == 0:
            return (ColumnNullType.NON_NULLABLE, None)
        else:
            return (ColumnNullType.USE_BITMASK, 0)

    @property
    def null_count(self) -> int:
        """
        Number of null elements, if known.

        Note: Arrow uses -1 to indicate "unknown", but None seems cleaner.
        """
        arrow_null_count = self._col.null_count
        n = arrow_null_count if arrow_null_count != -1 else None
        return n

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        The metadata for the column. See `DataFrame.metadata` for more details.
        """
        pass

    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.
        """
        return 1

    def get_chunks(self, n_chunks: Optional[int]=None) -> Iterable[_PyArrowColumn]:
        """
        Return an iterator yielding the chunks.

        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
        if n_chunks and n_chunks > 1:
            chunk_size = self.size() // n_chunks
            if self.size() % n_chunks != 0:
                chunk_size += 1
            array = self._col
            i = 0
            for start in range(0, chunk_size * n_chunks, chunk_size):
                yield _PyArrowColumn(array.slice(start, chunk_size), self._allow_copy)
                i += 1
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

    def _get_data_buffer(self) -> Tuple[_PyArrowBuffer, Any]:
        """
        Return the buffer containing the data and the buffer's
        associated dtype.
        """
        array = self._col
        dtype = self.dtype
        if pa.types.is_dictionary(array.type):
            array = array.indices
            dtype = _PyArrowColumn(array).dtype
        n = len(array.buffers())
        if n == 2:
            return (_PyArrowBuffer(array.buffers()[1]), dtype)
        elif n == 3:
            return (_PyArrowBuffer(array.buffers()[2]), dtype)

    def _get_validity_buffer(self) -> Tuple[_PyArrowBuffer, Any]:
        """
        Return the buffer containing the mask values indicating missing data
        and the buffer's associated dtype.
        Raises NoBufferPresent if null representation is not a bit or byte
        mask.
        """
        dtype = (DtypeKind.BOOL, 1, 'b', Endianness.NATIVE)
        array = self._col
        buff = array.buffers()[0]
        if buff:
            return (_PyArrowBuffer(buff), dtype)
        else:
            raise NoBufferPresent('There are no missing values so does not have a separate mask')

    def _get_offsets_buffer(self) -> Tuple[_PyArrowBuffer, Any]:
        """
        Return the buffer containing the offset values for variable-size binary
        data (e.g., variable-length strings) and the buffer's associated dtype.
        Raises NoBufferPresent if the data buffer does not have an associated
        offsets buffer.
        """
        array = self._col
        n = len(array.buffers())
        if n == 2:
            raise NoBufferPresent('This column has a fixed-length dtype so it does not have an offsets buffer')
        elif n == 3:
            dtype = self._col.type
            if pa.types.is_large_string(dtype):
                dtype = (DtypeKind.INT, 64, 'l', Endianness.NATIVE)
            else:
                dtype = (DtypeKind.INT, 32, 'i', Endianness.NATIVE)
            return (_PyArrowBuffer(array.buffers()[1]), dtype)