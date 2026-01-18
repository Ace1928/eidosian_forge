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
class ADBCDatabase(PandasSQL):
    """
    This class enables conversion between DataFrame and SQL databases
    using ADBC to handle DataBase abstraction.

    Parameters
    ----------
    con : adbc_driver_manager.dbapi.Connection
    """

    def __init__(self, con) -> None:
        self.con = con

    @contextmanager
    def run_transaction(self):
        with self.con.cursor() as cur:
            try:
                yield cur
            except Exception:
                self.con.rollback()
                raise
            self.con.commit()

    def execute(self, sql: str | Select | TextClause, params=None):
        if not isinstance(sql, str):
            raise TypeError('Query must be a string unless using sqlalchemy.')
        args = [] if params is None else [params]
        cur = self.con.cursor()
        try:
            cur.execute(sql, *args)
            return cur
        except Exception as exc:
            try:
                self.con.rollback()
            except Exception as inner_exc:
                ex = DatabaseError(f'Execution failed on sql: {sql}\n{exc}\nunable to rollback')
                raise ex from inner_exc
            ex = DatabaseError(f"Execution failed on sql '{sql}': {exc}")
            raise ex from exc

    def read_table(self, table_name: str, index_col: str | list[str] | None=None, coerce_float: bool=True, parse_dates=None, columns=None, schema: str | None=None, chunksize: int | None=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        coerce_float : bool, default True
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg}``, where the arg corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table.
        schema : string, default None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQL database object.
        chunksize : int, default None
            Raises NotImplementedError
        dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
            Back-end data type applied to the resultant :class:`DataFrame`
            (still experimental). Behaviour is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
              (default).
            * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
              DataFrame.

            .. versionadded:: 2.0

        Returns
        -------
        DataFrame

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        """
        if coerce_float is not True:
            raise NotImplementedError("'coerce_float' is not implemented for ADBC drivers")
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")
        if columns:
            if index_col:
                index_select = maybe_make_list(index_col)
            else:
                index_select = []
            to_select = index_select + columns
            select_list = ', '.join((f'"{x}"' for x in to_select))
        else:
            select_list = '*'
        if schema:
            stmt = f'SELECT {select_list} FROM {schema}.{table_name}'
        else:
            stmt = f'SELECT {select_list} FROM {table_name}'
        mapping: type[ArrowDtype] | None | Callable
        if dtype_backend == 'pyarrow':
            mapping = ArrowDtype
        elif dtype_backend == 'numpy_nullable':
            from pandas.io._util import _arrow_dtype_mapping
            mapping = _arrow_dtype_mapping().get
        elif using_pyarrow_string_dtype():
            from pandas.io._util import arrow_string_types_mapper
            arrow_string_types_mapper()
        else:
            mapping = None
        with self.con.cursor() as cur:
            cur.execute(stmt)
            df = cur.fetch_arrow_table().to_pandas(types_mapper=mapping)
        return _wrap_result_adbc(df, index_col=index_col, parse_dates=parse_dates)

    def read_query(self, sql: str, index_col: str | list[str] | None=None, coerce_float: bool=True, parse_dates=None, params=None, chunksize: int | None=None, dtype: DtypeArg | None=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Raises NotImplementedError
        params : list, tuple or dict, optional, default: None
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            Raises NotImplementedError
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {'a': np.float64, 'b': np.int32, 'c': 'Int64'}

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
        if coerce_float is not True:
            raise NotImplementedError("'coerce_float' is not implemented for ADBC drivers")
        if params:
            raise NotImplementedError("'params' is not implemented for ADBC drivers")
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")
        mapping: type[ArrowDtype] | None | Callable
        if dtype_backend == 'pyarrow':
            mapping = ArrowDtype
        elif dtype_backend == 'numpy_nullable':
            from pandas.io._util import _arrow_dtype_mapping
            mapping = _arrow_dtype_mapping().get
        else:
            mapping = None
        with self.con.cursor() as cur:
            cur.execute(sql)
            df = cur.fetch_arrow_table().to_pandas(types_mapper=mapping)
        return _wrap_result_adbc(df, index_col=index_col, parse_dates=parse_dates, dtype=dtype)
    read_sql = read_query

    def to_sql(self, frame, name: str, if_exists: Literal['fail', 'replace', 'append']='fail', index: bool=True, index_label=None, schema: str | None=None, chunksize: int | None=None, dtype: DtypeArg | None=None, method: Literal['multi'] | Callable | None=None, engine: str='auto', **engine_kwargs) -> int | None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame : DataFrame
        name : string
            Name of SQL table.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
        index : boolean, default True
            Write DataFrame index as a column.
        index_label : string or sequence, default None
            Raises NotImplementedError
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            Raises NotImplementedError
        dtype : single type or dict of column name to SQL type, default None
            Raises NotImplementedError
        method : {None', 'multi', callable}, default None
            Raises NotImplementedError
        engine : {'auto', 'sqlalchemy'}, default 'auto'
            Raises NotImplementedError if not set to 'auto'
        """
        if index_label:
            raise NotImplementedError("'index_label' is not implemented for ADBC drivers")
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")
        if dtype:
            raise NotImplementedError("'dtype' is not implemented for ADBC drivers")
        if method:
            raise NotImplementedError("'method' is not implemented for ADBC drivers")
        if engine != 'auto':
            raise NotImplementedError("engine != 'auto' not implemented for ADBC drivers")
        if schema:
            table_name = f'{schema}.{name}'
        else:
            table_name = name
        mode = 'create'
        if self.has_table(name, schema):
            if if_exists == 'fail':
                raise ValueError(f"Table '{table_name}' already exists.")
            elif if_exists == 'replace':
                with self.con.cursor() as cur:
                    cur.execute(f'DROP TABLE {table_name}')
            elif if_exists == 'append':
                mode = 'append'
        import pyarrow as pa
        try:
            tbl = pa.Table.from_pandas(frame, preserve_index=index)
        except pa.ArrowNotImplementedError as exc:
            raise ValueError('datatypes not supported') from exc
        with self.con.cursor() as cur:
            total_inserted = cur.adbc_ingest(table_name, tbl, mode=mode)
        self.con.commit()
        return total_inserted

    def has_table(self, name: str, schema: str | None=None) -> bool:
        meta = self.con.adbc_get_objects(db_schema_filter=schema, table_name_filter=name).read_all()
        for catalog_schema in meta['catalog_db_schemas'].to_pylist():
            if not catalog_schema:
                continue
            for schema_record in catalog_schema:
                if not schema_record:
                    continue
                for table_record in schema_record['db_schema_tables']:
                    if table_record['table_name'] == name:
                        return True
        return False

    def _create_sql_schema(self, frame: DataFrame, table_name: str, keys: list[str] | None=None, dtype: DtypeArg | None=None, schema: str | None=None) -> str:
        raise NotImplementedError('not implemented for adbc')