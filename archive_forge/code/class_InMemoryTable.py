import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
class InMemoryTable(TableBlock):
    """
    The table is said in-memory when it is loaded into the user's RAM.

    Pickling it does copy all the data using memory.
    Its implementation is simple and uses the underlying pyarrow Table methods directly.

    This is different from the `MemoryMapped` table, for which pickling doesn't copy all the
    data in memory. For a `MemoryMapped`, unpickling instead reloads the table from the disk.

    `InMemoryTable` must be used when data fit in memory, while `MemoryMapped` are reserved for
    data bigger than memory or when you want the memory footprint of your application to
    stay low.
    """

    @classmethod
    def from_file(cls, filename: str):
        table = _in_memory_arrow_table_from_file(filename)
        return cls(table)

    @classmethod
    def from_buffer(cls, buffer: pa.Buffer):
        table = _in_memory_arrow_table_from_buffer(buffer)
        return cls(table)

    @classmethod
    def from_pandas(cls, *args, **kwargs):
        """
        Convert pandas.DataFrame to an Arrow Table.

        The column types in the resulting Arrow Table are inferred from the
        dtypes of the pandas.Series in the DataFrame. In the case of non-object
        Series, the NumPy dtype is translated to its Arrow equivalent. In the
        case of `object`, we need to guess the datatype by looking at the
        Python objects in this Series.

        Be aware that Series of the `object` dtype don't carry enough
        information to always lead to a meaningful Arrow type. In the case that
        we cannot infer a type, e.g. because the DataFrame is of length 0 or
        the Series only contains `None/nan` objects, the type is set to
        null. This behavior can be avoided by constructing an explicit schema
        and passing it to this function.

        Args:
            df (`pandas.DataFrame`):
            schema (`pyarrow.Schema`, *optional*):
                The expected schema of the Arrow Table. This can be used to
                indicate the type of columns if we cannot infer it automatically.
                If passed, the output will have exactly this schema. Columns
                specified in the schema that are not found in the DataFrame columns
                or its index will raise an error. Additional columns or index
                levels in the DataFrame which are not specified in the schema will
                be ignored.
            preserve_index (`bool`, *optional*):
                Whether to store the index as an additional column in the resulting
                `Table`. The default of None will store the index as a column,
                except for RangeIndex which is stored as metadata only. Use
                `preserve_index=True` to force it to be stored as a column.
            nthreads (`int`, defaults to `None` (may use up to system CPU count threads))
                If greater than 1, convert columns to Arrow in parallel using
                indicated number of threads.
            columns (`List[str]`, *optional*):
               List of column to be converted. If `None`, use all columns.
            safe (`bool`, defaults to `True`):
               Check for overflows or other unsafe conversions,

        Returns:
            `datasets.table.Table`:

        Examples:
        ```python
        >>> import pandas as pd
        >>> import pyarrow as pa
        >>> df = pd.DataFrame({
            ...     'int': [1, 2],
            ...     'str': ['a', 'b']
            ... })
        >>> pa.Table.from_pandas(df)
        <pyarrow.lib.Table object at 0x7f05d1fb1b40>
        ```
        """
        return cls(pa.Table.from_pandas(*args, **kwargs))

    @classmethod
    def from_arrays(cls, *args, **kwargs):
        """
        Construct a Table from Arrow arrays.

        Args:
            arrays (`List[Union[pyarrow.Array, pyarrow.ChunkedArray]]`):
                Equal-length arrays that should form the table.
            names (`List[str]`, *optional*):
                Names for the table columns. If not passed, schema must be passed.
            schema (`Schema`, defaults to `None`):
                Schema for the created table. If not passed, names must be passed.
            metadata (`Union[dict, Mapping]`, defaults to `None`):
                Optional metadata for the schema (if inferred).

        Returns:
            `datasets.table.Table`
        """
        return cls(pa.Table.from_arrays(*args, **kwargs))

    @classmethod
    def from_pydict(cls, *args, **kwargs):
        """
        Construct a Table from Arrow arrays or columns.

        Args:
            mapping (`Union[dict, Mapping]`):
                A mapping of strings to Arrays or Python lists.
            schema (`Schema`, defaults to `None`):
                If not passed, will be inferred from the Mapping values
            metadata (`Union[dict, Mapping]`, defaults to `None`):
                Optional metadata for the schema (if inferred).

        Returns:
            `datasets.table.Table`
        """
        return cls(pa.Table.from_pydict(*args, **kwargs))

    @classmethod
    def from_pylist(cls, mapping, *args, **kwargs):
        """
        Construct a Table from list of rows / dictionaries.

        Args:
            mapping (`List[dict]`):
                A mapping of strings to row values.
            schema (`Schema`, defaults to `None`):
                If not passed, will be inferred from the Mapping values
            metadata (`Union[dict, Mapping]`, defaults to `None`):
                Optional metadata for the schema (if inferred).

        Returns:
            `datasets.table.Table`
        """
        return cls(pa.Table.from_pylist(mapping, *args, **kwargs))

    @classmethod
    def from_batches(cls, *args, **kwargs):
        """
        Construct a Table from a sequence or iterator of Arrow `RecordBatches`.

        Args:
            batches (`Union[Sequence[pyarrow.RecordBatch], Iterator[pyarrow.RecordBatch]]`):
                Sequence of `RecordBatch` to be converted, all schemas must be equal.
            schema (`Schema`, defaults to `None`):
                If not passed, will be inferred from the first `RecordBatch`.

        Returns:
            `datasets.table.Table`:
        """
        return cls(pa.Table.from_batches(*args, **kwargs))

    def slice(self, offset=0, length=None):
        """
        Compute zero-copy slice of this Table.

        Args:
            offset (`int`, defaults to `0`):
                Offset from start of table to slice.
            length (`int`, defaults to `None`):
                Length of slice (default is until end of table starting from
                offset).

        Returns:
            `datasets.table.Table`
        """
        return InMemoryTable(self.fast_slice(offset=offset, length=length))

    def filter(self, *args, **kwargs):
        """
        Select records from a Table. See `pyarrow.compute.filter` for full usage.
        """
        return InMemoryTable(self.table.filter(*args, **kwargs))

    def flatten(self, *args, **kwargs):
        """
        Flatten this Table.  Each column with a struct type is flattened
        into one column per struct field.  Other columns are left unchanged.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        return InMemoryTable(table_flatten(self.table, *args, **kwargs))

    def combine_chunks(self, *args, **kwargs):
        """
        Make a new table by combining the chunks this table has.

        All the underlying chunks in the `ChunkedArray` of each column are
        concatenated into zero or one chunk.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        return InMemoryTable(self.table.combine_chunks(*args, **kwargs))

    def cast(self, *args, **kwargs):
        """
        Cast table values to another schema.

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """
        return InMemoryTable(table_cast(self.table, *args, **kwargs))

    def replace_schema_metadata(self, *args, **kwargs):
        """
        EXPERIMENTAL: Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be `None`,
        which deletes any existing metadata).

        Args:
            metadata (`dict`, defaults to `None`):

        Returns:
            `datasets.table.Table`: shallow_copy
        """
        return InMemoryTable(self.table.replace_schema_metadata(*args, **kwargs))

    def add_column(self, *args, **kwargs):
        """
        Add column to Table at position.

        A new table is returned with the column added, the original table
        object is left unchanged.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column added.
        """
        return InMemoryTable(self.table.add_column(*args, **kwargs))

    def append_column(self, *args, **kwargs):
        """
        Append column at end of columns.

        Args:
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column added.
        """
        return InMemoryTable(self.table.append_column(*args, **kwargs))

    def remove_column(self, *args, **kwargs):
        """
        Create new Table with the indicated column removed.

        Args:
            i (`int`):
                Index of column to remove.

        Returns:
            `datasets.table.Table`:
                New table without the column.
        """
        return InMemoryTable(self.table.remove_column(*args, **kwargs))

    def set_column(self, *args, **kwargs):
        """
        Replace column in Table at position.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column set.
        """
        return InMemoryTable(self.table.set_column(*args, **kwargs))

    def rename_columns(self, *args, **kwargs):
        """
        Create new table with columns renamed to provided names.
        """
        return InMemoryTable(self.table.rename_columns(*args, **kwargs))

    def drop(self, *args, **kwargs):
        """
        Drop one or more columns and return a new table.

        Args:
            columns (`List[str]`):
                List of field names referencing existing columns.

        Raises:
            `KeyError` : if any of the passed columns name are not existing.

        Returns:
            `datasets.table.Table`:
                New table without the columns.
        """
        return InMemoryTable(self.table.drop(*args, **kwargs))

    def select(self, *args, **kwargs):
        """
        Select columns of the table.

        Returns a new table with the specified columns, and metadata preserved.

        Args:
            columns (:obj:`Union[List[str], List[int]]`):
                The column names or integer indices to select.

        Returns:
            :class:`datasets.table.Table`: New table with the specified columns, and metadata preserved.
        """
        return InMemoryTable(self.table.select(*args, **kwargs))