from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
class SQLTable(PandasObject):
    """
    For mapping Pandas tables to SQL tables.
    Uses fact that table is reflected by SQLAlchemy to
    do better type conversions.
    Also holds various flags needed to avoid having to
    pass them between functions all the time.
    """

    def __init__(self, name: str, pandas_sql_engine, frame=None, index: bool | str | list[str] | None=True, if_exists: Literal['fail', 'replace', 'append']='fail', prefix: str='pandas', index_label=None, schema=None, keys=None, dtype: DtypeArg | None=None) -> None:
        self.name = name
        self.pd_sql = pandas_sql_engine
        self.prefix = prefix
        self.frame = frame
        self.index = self._index_name(index, index_label)
        self.schema = schema
        self.if_exists = if_exists
        self.keys = keys
        self.dtype = dtype
        if frame is not None:
            self.table = self._create_table_setup()
        else:
            self.table = self.pd_sql.get_table(self.name, self.schema)
        if self.table is None:
            raise ValueError(f"Could not init table '{name}'")
        if not len(self.name):
            raise ValueError('Empty table name specified')

    def exists(self):
        return self.pd_sql.has_table(self.name, self.schema)

    def sql_schema(self) -> str:
        from sqlalchemy.schema import CreateTable
        return str(CreateTable(self.table).compile(self.pd_sql.con))

    def _execute_create(self) -> None:
        self.table = self.table.to_metadata(self.pd_sql.meta)
        with self.pd_sql.run_transaction():
            self.table.create(bind=self.pd_sql.con)

    def create(self) -> None:
        if self.exists():
            if self.if_exists == 'fail':
                raise ValueError(f"Table '{self.name}' already exists.")
            if self.if_exists == 'replace':
                self.pd_sql.drop_table(self.name, self.schema)
                self._execute_create()
            elif self.if_exists == 'append':
                pass
            else:
                raise ValueError(f"'{self.if_exists}' is not valid for if_exists")
        else:
            self._execute_create()

    def _execute_insert(self, conn, keys: list[str], data_iter) -> int:
        """
        Execute SQL statement inserting data

        Parameters
        ----------
        conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
        keys : list of str
           Column names
        data_iter : generator of list
           Each item contains a list of values to be inserted
        """
        data = [dict(zip(keys, row)) for row in data_iter]
        result = conn.execute(self.table.insert(), data)
        return result.rowcount

    def _execute_insert_multi(self, conn, keys: list[str], data_iter) -> int:
        """
        Alternative to _execute_insert for DBs support multi-value INSERT.

        Note: multi-value insert is usually faster for analytics DBs
        and tables containing a few columns
        but performance degrades quickly with increase of columns.

        """
        from sqlalchemy import insert
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(self.table).values(data)
        result = conn.execute(stmt)
        return result.rowcount

    def insert_data(self) -> tuple[list[str], list[np.ndarray]]:
        if self.index is not None:
            temp = self.frame.copy()
            temp.index.names = self.index
            try:
                temp.reset_index(inplace=True)
            except ValueError as err:
                raise ValueError(f'duplicate name in index/columns: {err}') from err
        else:
            temp = self.frame
        column_names = list(map(str, temp.columns))
        ncols = len(column_names)
        data_list: list[np.ndarray] = [None] * ncols
        for i, (_, ser) in enumerate(temp.items()):
            if ser.dtype.kind == 'M':
                if isinstance(ser._values, ArrowExtensionArray):
                    import pyarrow as pa
                    if pa.types.is_date(ser.dtype.pyarrow_dtype):
                        d = ser._values.to_numpy(dtype=object)
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=FutureWarning)
                            d = np.asarray(ser.dt.to_pydatetime(), dtype=object)
                else:
                    d = ser._values.to_pydatetime()
            elif ser.dtype.kind == 'm':
                vals = ser._values
                if isinstance(vals, ArrowExtensionArray):
                    vals = vals.to_numpy(dtype=np.dtype('m8[ns]'))
                d = vals.view('i8').astype(object)
            else:
                d = ser._values.astype(object)
            assert isinstance(d, np.ndarray), type(d)
            if ser._can_hold_na:
                mask = isna(d)
                d[mask] = None
            data_list[i] = d
        return (column_names, data_list)

    def insert(self, chunksize: int | None=None, method: Literal['multi'] | Callable | None=None) -> int | None:
        if method is None:
            exec_insert = self._execute_insert
        elif method == 'multi':
            exec_insert = self._execute_insert_multi
        elif callable(method):
            exec_insert = partial(method, self)
        else:
            raise ValueError(f'Invalid parameter `method`: {method}')
        keys, data_list = self.insert_data()
        nrows = len(self.frame)
        if nrows == 0:
            return 0
        if chunksize is None:
            chunksize = nrows
        elif chunksize == 0:
            raise ValueError('chunksize argument should be non-zero')
        chunks = nrows // chunksize + 1
        total_inserted = None
        with self.pd_sql.run_transaction() as conn:
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, nrows)
                if start_i >= end_i:
                    break
                chunk_iter = zip(*(arr[start_i:end_i] for arr in data_list))
                num_inserted = exec_insert(conn, keys, chunk_iter)
                if num_inserted is not None:
                    if total_inserted is None:
                        total_inserted = num_inserted
                    else:
                        total_inserted += num_inserted
        return total_inserted

    def _query_iterator(self, result, exit_stack: ExitStack, chunksize: int | None, columns, coerce_float: bool=True, parse_dates=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy'):
        """Return generator through chunked result set."""
        has_read_data = False
        with exit_stack:
            while True:
                data = result.fetchmany(chunksize)
                if not data:
                    if not has_read_data:
                        yield DataFrame.from_records([], columns=columns, coerce_float=coerce_float)
                    break
                has_read_data = True
                self.frame = _convert_arrays_to_dataframe(data, columns, coerce_float, dtype_backend)
                self._harmonize_columns(parse_dates=parse_dates, dtype_backend=dtype_backend)
                if self.index is not None:
                    self.frame.set_index(self.index, inplace=True)
                yield self.frame

    def read(self, exit_stack: ExitStack, coerce_float: bool=True, parse_dates=None, columns=None, chunksize: int | None=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> DataFrame | Iterator[DataFrame]:
        from sqlalchemy import select
        if columns is not None and len(columns) > 0:
            cols = [self.table.c[n] for n in columns]
            if self.index is not None:
                for idx in self.index[::-1]:
                    cols.insert(0, self.table.c[idx])
            sql_select = select(*cols)
        else:
            sql_select = select(self.table)
        result = self.pd_sql.execute(sql_select)
        column_names = result.keys()
        if chunksize is not None:
            return self._query_iterator(result, exit_stack, chunksize, column_names, coerce_float=coerce_float, parse_dates=parse_dates, dtype_backend=dtype_backend)
        else:
            data = result.fetchall()
            self.frame = _convert_arrays_to_dataframe(data, column_names, coerce_float, dtype_backend)
            self._harmonize_columns(parse_dates=parse_dates, dtype_backend=dtype_backend)
            if self.index is not None:
                self.frame.set_index(self.index, inplace=True)
            return self.frame

    def _index_name(self, index, index_label):
        if index is True:
            nlevels = self.frame.index.nlevels
            if index_label is not None:
                if not isinstance(index_label, list):
                    index_label = [index_label]
                if len(index_label) != nlevels:
                    raise ValueError(f"Length of 'index_label' should match number of levels, which is {nlevels}")
                return index_label
            if nlevels == 1 and 'index' not in self.frame.columns and (self.frame.index.name is None):
                return ['index']
            else:
                return com.fill_missing_names(self.frame.index.names)
        elif isinstance(index, str):
            return [index]
        elif isinstance(index, list):
            return index
        else:
            return None

    def _get_column_names_and_types(self, dtype_mapper):
        column_names_and_types = []
        if self.index is not None:
            for i, idx_label in enumerate(self.index):
                idx_type = dtype_mapper(self.frame.index._get_level_values(i))
                column_names_and_types.append((str(idx_label), idx_type, True))
        column_names_and_types += [(str(self.frame.columns[i]), dtype_mapper(self.frame.iloc[:, i]), False) for i in range(len(self.frame.columns))]
        return column_names_and_types

    def _create_table_setup(self):
        from sqlalchemy import Column, PrimaryKeyConstraint, Table
        from sqlalchemy.schema import MetaData
        column_names_and_types = self._get_column_names_and_types(self._sqlalchemy_type)
        columns: list[Any] = [Column(name, typ, index=is_index) for name, typ, is_index in column_names_and_types]
        if self.keys is not None:
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            pkc = PrimaryKeyConstraint(*keys, name=self.name + '_pk')
            columns.append(pkc)
        schema = self.schema or self.pd_sql.meta.schema
        meta = MetaData()
        return Table(self.name, meta, *columns, schema=schema)

    def _harmonize_columns(self, parse_dates=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> None:
        """
        Make the DataFrame's column types align with the SQL table
        column types.
        Need to work around limited NA value support. Floats are always
        fine, ints must always be floats if there are Null values.
        Booleans are hard because converting bool column with None replaces
        all Nones with false. Therefore only convert bool if there are no
        NA values.
        Datetimes should already be converted to np.datetime64 if supported,
        but here we also force conversion if required.
        """
        parse_dates = _process_parse_dates_argument(parse_dates)
        for sql_col in self.table.columns:
            col_name = sql_col.name
            try:
                df_col = self.frame[col_name]
                if col_name in parse_dates:
                    try:
                        fmt = parse_dates[col_name]
                    except TypeError:
                        fmt = None
                    self.frame[col_name] = _handle_date_column(df_col, format=fmt)
                    continue
                col_type = self._get_dtype(sql_col.type)
                if col_type is datetime or col_type is date or col_type is DatetimeTZDtype:
                    utc = col_type is DatetimeTZDtype
                    self.frame[col_name] = _handle_date_column(df_col, utc=utc)
                elif dtype_backend == 'numpy' and col_type is float:
                    self.frame[col_name] = df_col.astype(col_type, copy=False)
                elif dtype_backend == 'numpy' and len(df_col) == df_col.count():
                    if col_type is np.dtype('int64') or col_type is bool:
                        self.frame[col_name] = df_col.astype(col_type, copy=False)
            except KeyError:
                pass

    def _sqlalchemy_type(self, col: Index | Series):
        dtype: DtypeArg = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if col.name in dtype:
                return dtype[col.name]
        col_type = lib.infer_dtype(col, skipna=True)
        from sqlalchemy.types import TIMESTAMP, BigInteger, Boolean, Date, DateTime, Float, Integer, SmallInteger, Text, Time
        if col_type in ('datetime64', 'datetime'):
            try:
                if col.dt.tz is not None:
                    return TIMESTAMP(timezone=True)
            except AttributeError:
                if getattr(col, 'tz', None) is not None:
                    return TIMESTAMP(timezone=True)
            return DateTime
        if col_type == 'timedelta64':
            warnings.warn("the 'timedelta' type is not supported, and will be written as integer values (ns frequency) to the database.", UserWarning, stacklevel=find_stack_level())
            return BigInteger
        elif col_type == 'floating':
            if col.dtype == 'float32':
                return Float(precision=23)
            else:
                return Float(precision=53)
        elif col_type == 'integer':
            if col.dtype.name.lower() in ('int8', 'uint8', 'int16'):
                return SmallInteger
            elif col.dtype.name.lower() in ('uint16', 'int32'):
                return Integer
            elif col.dtype.name.lower() == 'uint64':
                raise ValueError('Unsigned 64 bit integer datatype is not supported')
            else:
                return BigInteger
        elif col_type == 'boolean':
            return Boolean
        elif col_type == 'date':
            return Date
        elif col_type == 'time':
            return Time
        elif col_type == 'complex':
            raise ValueError('Complex datatypes not supported')
        return Text

    def _get_dtype(self, sqltype):
        from sqlalchemy.types import TIMESTAMP, Boolean, Date, DateTime, Float, Integer
        if isinstance(sqltype, Float):
            return float
        elif isinstance(sqltype, Integer):
            return np.dtype('int64')
        elif isinstance(sqltype, TIMESTAMP):
            if not sqltype.timezone:
                return datetime
            return DatetimeTZDtype
        elif isinstance(sqltype, DateTime):
            return datetime
        elif isinstance(sqltype, Date):
            return date
        elif isinstance(sqltype, Boolean):
            return bool
        return object