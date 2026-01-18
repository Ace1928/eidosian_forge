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
class SQLDatabase(PandasSQL):
    """
    This class enables conversion between DataFrame and SQL databases
    using SQLAlchemy to handle DataBase abstraction.

    Parameters
    ----------
    con : SQLAlchemy Connectable or URI string.
        Connectable to connect with the database. Using SQLAlchemy makes it
        possible to use any DB supported by that library.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    need_transaction : bool, default False
        If True, SQLDatabase will create a transaction.

    """

    def __init__(self, con, schema: str | None=None, need_transaction: bool=False) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine
        from sqlalchemy.schema import MetaData
        self.exit_stack = ExitStack()
        if isinstance(con, str):
            con = create_engine(con)
            self.exit_stack.callback(con.dispose)
        if isinstance(con, Engine):
            con = self.exit_stack.enter_context(con.connect())
        if need_transaction and (not con.in_transaction()):
            self.exit_stack.enter_context(con.begin())
        self.con = con
        self.meta = MetaData(schema=schema)
        self.returns_generator = False

    def __exit__(self, *args) -> None:
        if not self.returns_generator:
            self.exit_stack.close()

    @contextmanager
    def run_transaction(self):
        if not self.con.in_transaction():
            with self.con.begin():
                yield self.con
        else:
            yield self.con

    def execute(self, sql: str | Select | TextClause, params=None):
        """Simple passthrough to SQLAlchemy connectable"""
        args = [] if params is None else [params]
        if isinstance(sql, str):
            return self.con.exec_driver_sql(sql, *args)
        return self.con.execute(sql, *args)

    def read_table(self, table_name: str, index_col: str | list[str] | None=None, coerce_float: bool=True, parse_dates=None, columns=None, schema: str | None=None, chunksize: int | None=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        index_col : string, optional, default: None
            Column to set as index.
        coerce_float : bool, default True
            Attempts to convert values of non-string, non-numeric objects
            (like decimal.Decimal) to floating point. This can result in
            loss of precision.
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
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
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
        self.meta.reflect(bind=self.con, only=[table_name], views=True)
        table = SQLTable(table_name, self, index=index_col, schema=schema)
        if chunksize is not None:
            self.returns_generator = True
        return table.read(self.exit_stack, coerce_float=coerce_float, parse_dates=parse_dates, columns=columns, chunksize=chunksize, dtype_backend=dtype_backend)

    @staticmethod
    def _query_iterator(result, exit_stack: ExitStack, chunksize: int, columns, index_col=None, coerce_float: bool=True, parse_dates=None, dtype: DtypeArg | None=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy'):
        """Return generator through chunked result set"""
        has_read_data = False
        with exit_stack:
            while True:
                data = result.fetchmany(chunksize)
                if not data:
                    if not has_read_data:
                        yield _wrap_result([], columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype, dtype_backend=dtype_backend)
                    break
                has_read_data = True
                yield _wrap_result(data, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype, dtype_backend=dtype_backend)

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
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method.  The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
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
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
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
        result = self.execute(sql, params)
        columns = result.keys()
        if chunksize is not None:
            self.returns_generator = True
            return self._query_iterator(result, self.exit_stack, chunksize, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype, dtype_backend=dtype_backend)
        else:
            data = result.fetchall()
            frame = _wrap_result(data, columns, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, dtype=dtype, dtype_backend=dtype_backend)
            return frame
    read_sql = read_query

    def prep_table(self, frame, name: str, if_exists: Literal['fail', 'replace', 'append']='fail', index: bool | str | list[str] | None=True, index_label=None, schema=None, dtype: DtypeArg | None=None) -> SQLTable:
        """
        Prepares table in the database for data insertion. Creates it if needed, etc.
        """
        if dtype:
            if not is_dict_like(dtype):
                dtype = {col_name: dtype for col_name in frame}
            else:
                dtype = cast(dict, dtype)
            from sqlalchemy.types import TypeEngine
            for col, my_type in dtype.items():
                if isinstance(my_type, type) and issubclass(my_type, TypeEngine):
                    pass
                elif isinstance(my_type, TypeEngine):
                    pass
                else:
                    raise ValueError(f'The type of {col} is not a SQLAlchemy type')
        table = SQLTable(name, self, frame=frame, index=index, if_exists=if_exists, index_label=index_label, schema=schema, dtype=dtype)
        table.create()
        return table

    def check_case_sensitive(self, name: str, schema: str | None) -> None:
        """
        Checks table name for issues with case-sensitivity.
        Method is called after data is inserted.
        """
        if not name.isdigit() and (not name.islower()):
            from sqlalchemy import inspect as sqlalchemy_inspect
            insp = sqlalchemy_inspect(self.con)
            table_names = insp.get_table_names(schema=schema or self.meta.schema)
            if name not in table_names:
                msg = f"The provided table name '{name}' is not found exactly as such in the database after writing the table, possibly due to case sensitivity issues. Consider using lower case table names."
                warnings.warn(msg, UserWarning, stacklevel=find_stack_level())

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
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            If not None, then rows will be written in batches of this size at a
            time.  If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a SQLAlchemy type. If all columns are of the same type, one
            single value can be used.
        method : {None', 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        engine : {'auto', 'sqlalchemy'}, default 'auto'
            SQL engine library to use. If 'auto', then the option
            ``io.sql.engine`` is used. The default ``io.sql.engine``
            behavior is 'sqlalchemy'

            .. versionadded:: 1.3.0

        **engine_kwargs
            Any additional kwargs are passed to the engine.
        """
        sql_engine = get_engine(engine)
        table = self.prep_table(frame=frame, name=name, if_exists=if_exists, index=index, index_label=index_label, schema=schema, dtype=dtype)
        total_inserted = sql_engine.insert_records(table=table, con=self.con, frame=frame, name=name, index=index, schema=schema, chunksize=chunksize, method=method, **engine_kwargs)
        self.check_case_sensitive(name=name, schema=schema)
        return total_inserted

    @property
    def tables(self):
        return self.meta.tables

    def has_table(self, name: str, schema: str | None=None) -> bool:
        from sqlalchemy import inspect as sqlalchemy_inspect
        insp = sqlalchemy_inspect(self.con)
        return insp.has_table(name, schema or self.meta.schema)

    def get_table(self, table_name: str, schema: str | None=None) -> Table:
        from sqlalchemy import Numeric, Table
        schema = schema or self.meta.schema
        tbl = Table(table_name, self.meta, autoload_with=self.con, schema=schema)
        for column in tbl.columns:
            if isinstance(column.type, Numeric):
                column.type.asdecimal = False
        return tbl

    def drop_table(self, table_name: str, schema: str | None=None) -> None:
        schema = schema or self.meta.schema
        if self.has_table(table_name, schema):
            self.meta.reflect(bind=self.con, only=[table_name], schema=schema, views=True)
            with self.run_transaction():
                self.get_table(table_name, schema).drop(bind=self.con)
            self.meta.clear()

    def _create_sql_schema(self, frame: DataFrame, table_name: str, keys: list[str] | None=None, dtype: DtypeArg | None=None, schema: str | None=None) -> str:
        table = SQLTable(table_name, self, frame=frame, index=False, keys=keys, dtype=dtype, schema=schema)
        return str(table.sql_schema())