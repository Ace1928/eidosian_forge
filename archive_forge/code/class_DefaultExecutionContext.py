from __future__ import annotations
import functools
import operator
import random
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import characteristics
from . import cursor as _cursor
from . import interfaces
from .base import Connection
from .interfaces import CacheStats
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .reflection import ObjectKind
from .reflection import ObjectScope
from .. import event
from .. import exc
from .. import pool
from .. import util
from ..sql import compiler
from ..sql import dml
from ..sql import expression
from ..sql import type_api
from ..sql._typing import is_tuple_type
from ..sql.base import _NoArg
from ..sql.compiler import DDLCompiler
from ..sql.compiler import InsertmanyvaluesSentinelOpts
from ..sql.compiler import SQLCompiler
from ..sql.elements import quoted_name
from ..util.typing import Final
from ..util.typing import Literal
class DefaultExecutionContext(ExecutionContext):
    isinsert = False
    isupdate = False
    isdelete = False
    is_crud = False
    is_text = False
    isddl = False
    execute_style: ExecuteStyle = ExecuteStyle.EXECUTE
    compiled: Optional[Compiled] = None
    result_column_struct: Optional[Tuple[List[ResultColumnsEntry], bool, bool, bool, bool]] = None
    returned_default_rows: Optional[Sequence[Row[Any]]] = None
    execution_options: _ExecuteOptions = util.EMPTY_DICT
    cursor_fetch_strategy = _cursor._DEFAULT_FETCH
    invoked_statement: Optional[Executable] = None
    _is_implicit_returning = False
    _is_explicit_returning = False
    _is_supplemental_returning = False
    _is_server_side = False
    _soft_closed = False
    _rowcount: Optional[int] = None
    _translate_colname: Optional[Callable[[str], str]] = None
    _expanded_parameters: Mapping[str, List[str]] = util.immutabledict()
    'used by set_input_sizes().\n\n    This collection comes from ``ExpandedState.parameter_expansion``.\n\n    '
    cache_hit = NO_CACHE_KEY
    root_connection: Connection
    _dbapi_connection: PoolProxiedConnection
    dialect: Dialect
    unicode_statement: str
    cursor: DBAPICursor
    compiled_parameters: List[_MutableCoreSingleExecuteParams]
    parameters: _DBAPIMultiExecuteParams
    extracted_parameters: Optional[Sequence[BindParameter[Any]]]
    _empty_dict_params = cast('Mapping[str, Any]', util.EMPTY_DICT)
    _insertmanyvalues_rows: Optional[List[Tuple[Any, ...]]] = None
    _num_sentinel_cols: int = 0

    @classmethod
    def _init_ddl(cls, dialect: Dialect, connection: Connection, dbapi_connection: PoolProxiedConnection, execution_options: _ExecuteOptions, compiled_ddl: DDLCompiler) -> ExecutionContext:
        """Initialize execution context for an ExecutableDDLElement
        construct."""
        self = cls.__new__(cls)
        self.root_connection = connection
        self._dbapi_connection = dbapi_connection
        self.dialect = connection.dialect
        self.compiled = compiled = compiled_ddl
        self.isddl = True
        self.execution_options = execution_options
        self.unicode_statement = str(compiled)
        if compiled.schema_translate_map:
            schema_translate_map = self.execution_options.get('schema_translate_map', {})
            rst = compiled.preparer._render_schema_translates
            self.unicode_statement = rst(self.unicode_statement, schema_translate_map)
        self.statement = self.unicode_statement
        self.cursor = self.create_cursor()
        self.compiled_parameters = []
        if dialect.positional:
            self.parameters = [dialect.execute_sequence_format()]
        else:
            self.parameters = [self._empty_dict_params]
        return self

    @classmethod
    def _init_compiled(cls, dialect: Dialect, connection: Connection, dbapi_connection: PoolProxiedConnection, execution_options: _ExecuteOptions, compiled: SQLCompiler, parameters: _CoreMultiExecuteParams, invoked_statement: Executable, extracted_parameters: Optional[Sequence[BindParameter[Any]]], cache_hit: CacheStats=CacheStats.CACHING_DISABLED) -> ExecutionContext:
        """Initialize execution context for a Compiled construct."""
        self = cls.__new__(cls)
        self.root_connection = connection
        self._dbapi_connection = dbapi_connection
        self.dialect = connection.dialect
        self.extracted_parameters = extracted_parameters
        self.invoked_statement = invoked_statement
        self.compiled = compiled
        self.cache_hit = cache_hit
        self.execution_options = execution_options
        self.result_column_struct = (compiled._result_columns, compiled._ordered_columns, compiled._textual_ordered_columns, compiled._ad_hoc_textual, compiled._loose_column_name_matching)
        self.isinsert = ii = compiled.isinsert
        self.isupdate = iu = compiled.isupdate
        self.isdelete = id_ = compiled.isdelete
        self.is_text = compiled.isplaintext
        if ii or iu or id_:
            dml_statement = compiled.compile_state.statement
            if TYPE_CHECKING:
                assert isinstance(dml_statement, UpdateBase)
            self.is_crud = True
            self._is_explicit_returning = ier = bool(dml_statement._returning)
            self._is_implicit_returning = iir = bool(compiled.implicit_returning)
            if iir and dml_statement._supplemental_returning:
                self._is_supplemental_returning = True
            assert not (iir and ier)
            if (ier or iir) and compiled.for_executemany:
                if ii and (not self.dialect.insert_executemany_returning):
                    raise exc.InvalidRequestError(f'Dialect {self.dialect.dialect_description} with current server capabilities does not support INSERT..RETURNING when executemany is used')
                elif ii and dml_statement._sort_by_parameter_order and (not self.dialect.insert_executemany_returning_sort_by_parameter_order):
                    raise exc.InvalidRequestError(f'Dialect {self.dialect.dialect_description} with current server capabilities does not support INSERT..RETURNING with deterministic row ordering when executemany is used')
                elif ii and self.dialect.use_insertmanyvalues and (not compiled._insertmanyvalues):
                    raise exc.InvalidRequestError('Statement does not have "insertmanyvalues" enabled, can\'t use INSERT..RETURNING with executemany in this case.')
                elif iu and (not self.dialect.update_executemany_returning):
                    raise exc.InvalidRequestError(f'Dialect {self.dialect.dialect_description} with current server capabilities does not support UPDATE..RETURNING when executemany is used')
                elif id_ and (not self.dialect.delete_executemany_returning):
                    raise exc.InvalidRequestError(f'Dialect {self.dialect.dialect_description} with current server capabilities does not support DELETE..RETURNING when executemany is used')
        if not parameters:
            self.compiled_parameters = [compiled.construct_params(extracted_parameters=extracted_parameters, escape_names=False)]
        else:
            self.compiled_parameters = [compiled.construct_params(m, escape_names=False, _group_number=grp, extracted_parameters=extracted_parameters) for grp, m in enumerate(parameters)]
            if len(parameters) > 1:
                if self.isinsert and compiled._insertmanyvalues:
                    self.execute_style = ExecuteStyle.INSERTMANYVALUES
                    imv = compiled._insertmanyvalues
                    if imv.sentinel_columns is not None:
                        self._num_sentinel_cols = imv.num_sentinel_columns
                else:
                    self.execute_style = ExecuteStyle.EXECUTEMANY
        self.unicode_statement = compiled.string
        self.cursor = self.create_cursor()
        if self.compiled.insert_prefetch or self.compiled.update_prefetch:
            self._process_execute_defaults()
        processors = compiled._bind_processors
        flattened_processors: Mapping[str, _BindProcessorType[Any]] = processors
        if compiled.literal_execute_params or compiled.post_compile_params:
            if self.executemany:
                raise exc.InvalidRequestError("'literal_execute' or 'expanding' parameters can't be used with executemany()")
            expanded_state = compiled._process_parameters_for_postcompile(self.compiled_parameters[0])
            self.unicode_statement = expanded_state.statement
            self._expanded_parameters = expanded_state.parameter_expansion
            flattened_processors = dict(processors)
            flattened_processors.update(expanded_state.processors)
            positiontup = expanded_state.positiontup
        elif compiled.positional:
            positiontup = self.compiled.positiontup
        else:
            positiontup = None
        if compiled.schema_translate_map:
            schema_translate_map = self.execution_options.get('schema_translate_map', {})
            rst = compiled.preparer._render_schema_translates
            self.unicode_statement = rst(self.unicode_statement, schema_translate_map)
        self.statement = self.unicode_statement
        if compiled.positional:
            core_positional_parameters: MutableSequence[Sequence[Any]] = []
            assert positiontup is not None
            for compiled_params in self.compiled_parameters:
                l_param: List[Any] = [flattened_processors[key](compiled_params[key]) if key in flattened_processors else compiled_params[key] for key in positiontup]
                core_positional_parameters.append(dialect.execute_sequence_format(l_param))
            self.parameters = core_positional_parameters
        else:
            core_dict_parameters: MutableSequence[Dict[str, Any]] = []
            escaped_names = compiled.escaped_bind_names
            d_param: Dict[str, Any]
            for compiled_params in self.compiled_parameters:
                if escaped_names:
                    d_param = {escaped_names.get(key, key): flattened_processors[key](compiled_params[key]) if key in flattened_processors else compiled_params[key] for key in compiled_params}
                else:
                    d_param = {key: flattened_processors[key](compiled_params[key]) if key in flattened_processors else compiled_params[key] for key in compiled_params}
                core_dict_parameters.append(d_param)
            self.parameters = core_dict_parameters
        return self

    @classmethod
    def _init_statement(cls, dialect: Dialect, connection: Connection, dbapi_connection: PoolProxiedConnection, execution_options: _ExecuteOptions, statement: str, parameters: _DBAPIMultiExecuteParams) -> ExecutionContext:
        """Initialize execution context for a string SQL statement."""
        self = cls.__new__(cls)
        self.root_connection = connection
        self._dbapi_connection = dbapi_connection
        self.dialect = connection.dialect
        self.is_text = True
        self.execution_options = execution_options
        if not parameters:
            if self.dialect.positional:
                self.parameters = [dialect.execute_sequence_format()]
            else:
                self.parameters = [self._empty_dict_params]
        elif isinstance(parameters[0], dialect.execute_sequence_format):
            self.parameters = parameters
        elif isinstance(parameters[0], dict):
            self.parameters = parameters
        else:
            self.parameters = [dialect.execute_sequence_format(p) for p in parameters]
        if len(parameters) > 1:
            self.execute_style = ExecuteStyle.EXECUTEMANY
        self.statement = self.unicode_statement = statement
        self.cursor = self.create_cursor()
        return self

    @classmethod
    def _init_default(cls, dialect: Dialect, connection: Connection, dbapi_connection: PoolProxiedConnection, execution_options: _ExecuteOptions) -> ExecutionContext:
        """Initialize execution context for a ColumnDefault construct."""
        self = cls.__new__(cls)
        self.root_connection = connection
        self._dbapi_connection = dbapi_connection
        self.dialect = connection.dialect
        self.execution_options = execution_options
        self.cursor = self.create_cursor()
        return self

    def _get_cache_stats(self) -> str:
        if self.compiled is None:
            return 'raw sql'
        now = perf_counter()
        ch = self.cache_hit
        gen_time = self.compiled._gen_time
        assert gen_time is not None
        if ch is NO_CACHE_KEY:
            return 'no key %.5fs' % (now - gen_time,)
        elif ch is CACHE_HIT:
            return 'cached since %.4gs ago' % (now - gen_time,)
        elif ch is CACHE_MISS:
            return 'generated in %.5fs' % (now - gen_time,)
        elif ch is CACHING_DISABLED:
            if '_cache_disable_reason' in self.execution_options:
                return 'caching disabled (%s) %.5fs ' % (self.execution_options['_cache_disable_reason'], now - gen_time)
            else:
                return 'caching disabled %.5fs' % (now - gen_time,)
        elif ch is NO_DIALECT_SUPPORT:
            return 'dialect %s+%s does not support caching %.5fs' % (self.dialect.name, self.dialect.driver, now - gen_time)
        else:
            return 'unknown'

    @property
    def executemany(self):
        return self.execute_style in (ExecuteStyle.EXECUTEMANY, ExecuteStyle.INSERTMANYVALUES)

    @util.memoized_property
    def identifier_preparer(self):
        if self.compiled:
            return self.compiled.preparer
        elif 'schema_translate_map' in self.execution_options:
            return self.dialect.identifier_preparer._with_schema_translate(self.execution_options['schema_translate_map'])
        else:
            return self.dialect.identifier_preparer

    @util.memoized_property
    def engine(self):
        return self.root_connection.engine

    @util.memoized_property
    def postfetch_cols(self) -> Optional[Sequence[Column[Any]]]:
        if TYPE_CHECKING:
            assert isinstance(self.compiled, SQLCompiler)
        return self.compiled.postfetch

    @util.memoized_property
    def prefetch_cols(self) -> Optional[Sequence[Column[Any]]]:
        if TYPE_CHECKING:
            assert isinstance(self.compiled, SQLCompiler)
        if self.isinsert:
            return self.compiled.insert_prefetch
        elif self.isupdate:
            return self.compiled.update_prefetch
        else:
            return ()

    @util.memoized_property
    def no_parameters(self):
        return self.execution_options.get('no_parameters', False)

    def _execute_scalar(self, stmt, type_, parameters=None):
        """Execute a string statement on the current cursor, returning a
        scalar result.

        Used to fire off sequences, default phrases, and "select lastrowid"
        types of statements individually or in the context of a parent INSERT
        or UPDATE statement.

        """
        conn = self.root_connection
        if 'schema_translate_map' in self.execution_options:
            schema_translate_map = self.execution_options.get('schema_translate_map', {})
            rst = self.identifier_preparer._render_schema_translates
            stmt = rst(stmt, schema_translate_map)
        if not parameters:
            if self.dialect.positional:
                parameters = self.dialect.execute_sequence_format()
            else:
                parameters = {}
        conn._cursor_execute(self.cursor, stmt, parameters, context=self)
        row = self.cursor.fetchone()
        if row is not None:
            r = row[0]
        else:
            r = None
        if type_ is not None:
            proc = type_._cached_result_processor(self.dialect, self.cursor.description[0][1])
            if proc:
                return proc(r)
        return r

    @util.memoized_property
    def connection(self):
        return self.root_connection

    def _use_server_side_cursor(self):
        if not self.dialect.supports_server_side_cursors:
            return False
        if self.dialect.server_side_cursors:
            use_server_side = self.execution_options.get('stream_results', True) and (self.compiled and isinstance(self.compiled.statement, expression.Selectable) or ((not self.compiled or isinstance(self.compiled.statement, expression.TextClause)) and self.unicode_statement and SERVER_SIDE_CURSOR_RE.match(self.unicode_statement)))
        else:
            use_server_side = self.execution_options.get('stream_results', False)
        return use_server_side

    def create_cursor(self):
        if self.dialect.supports_server_side_cursors and (self.execution_options.get('stream_results', False) or (self.dialect.server_side_cursors and self._use_server_side_cursor())):
            self._is_server_side = True
            return self.create_server_side_cursor()
        else:
            self._is_server_side = False
            return self.create_default_cursor()

    def fetchall_for_returning(self, cursor):
        return cursor.fetchall()

    def create_default_cursor(self):
        return self._dbapi_connection.cursor()

    def create_server_side_cursor(self):
        raise NotImplementedError()

    def pre_exec(self):
        pass

    def get_out_parameter_values(self, names):
        raise NotImplementedError('This dialect does not support OUT parameters')

    def post_exec(self):
        pass

    def get_result_processor(self, type_, colname, coltype):
        """Return a 'result processor' for a given type as present in
        cursor.description.

        This has a default implementation that dialects can override
        for context-sensitive result type handling.

        """
        return type_._cached_result_processor(self.dialect, coltype)

    def get_lastrowid(self):
        """return self.cursor.lastrowid, or equivalent, after an INSERT.

        This may involve calling special cursor functions, issuing a new SELECT
        on the cursor (or a new one), or returning a stored value that was
        calculated within post_exec().

        This function will only be called for dialects which support "implicit"
        primary key generation, keep preexecute_autoincrement_sequences set to
        False, and when no explicit id value was bound to the statement.

        The function is called once for an INSERT statement that would need to
        return the last inserted primary key for those dialects that make use
        of the lastrowid concept.  In these cases, it is called directly after
        :meth:`.ExecutionContext.post_exec`.

        """
        return self.cursor.lastrowid

    def handle_dbapi_exception(self, e):
        pass

    @util.non_memoized_property
    def rowcount(self) -> int:
        if self._rowcount is not None:
            return self._rowcount
        else:
            return self.cursor.rowcount

    @property
    def _has_rowcount(self):
        return self._rowcount is not None

    def supports_sane_rowcount(self):
        return self.dialect.supports_sane_rowcount

    def supports_sane_multi_rowcount(self):
        return self.dialect.supports_sane_multi_rowcount

    def _setup_result_proxy(self):
        exec_opt = self.execution_options
        if self._rowcount is None and exec_opt.get('preserve_rowcount', False):
            self._rowcount = self.cursor.rowcount
        if self.is_crud or self.is_text:
            result = self._setup_dml_or_text_result()
            yp = sr = False
        else:
            yp = exec_opt.get('yield_per', None)
            sr = self._is_server_side or exec_opt.get('stream_results', False)
            strategy = self.cursor_fetch_strategy
            if sr and strategy is _cursor._DEFAULT_FETCH:
                strategy = _cursor.BufferedRowCursorFetchStrategy(self.cursor, self.execution_options)
            cursor_description: _DBAPICursorDescription = strategy.alternate_cursor_description or self.cursor.description
            if cursor_description is None:
                strategy = _cursor._NO_CURSOR_DQL
            result = _cursor.CursorResult(self, strategy, cursor_description)
        compiled = self.compiled
        if compiled and (not self.isddl) and cast(SQLCompiler, compiled).has_out_parameters:
            self._setup_out_parameters(result)
        self._soft_closed = result._soft_closed
        if yp:
            result = result.yield_per(yp)
        return result

    def _setup_out_parameters(self, result):
        compiled = cast(SQLCompiler, self.compiled)
        out_bindparams = [(param, name) for param, name in compiled.bind_names.items() if param.isoutparam]
        out_parameters = {}
        for bindparam, raw_value in zip([param for param, name in out_bindparams], self.get_out_parameter_values([name for param, name in out_bindparams])):
            type_ = bindparam.type
            impl_type = type_.dialect_impl(self.dialect)
            dbapi_type = impl_type.get_dbapi_type(self.dialect.loaded_dbapi)
            result_processor = impl_type.result_processor(self.dialect, dbapi_type)
            if result_processor is not None:
                raw_value = result_processor(raw_value)
            out_parameters[bindparam.key] = raw_value
        result.out_parameters = out_parameters

    def _setup_dml_or_text_result(self):
        compiled = cast(SQLCompiler, self.compiled)
        strategy: ResultFetchStrategy = self.cursor_fetch_strategy
        if self.isinsert:
            if self.execute_style is ExecuteStyle.INSERTMANYVALUES and compiled.effective_returning:
                strategy = _cursor.FullyBufferedCursorFetchStrategy(self.cursor, initial_buffer=self._insertmanyvalues_rows, alternate_description=strategy.alternate_cursor_description)
            if compiled.postfetch_lastrowid:
                self.inserted_primary_key_rows = self._setup_ins_pk_from_lastrowid()
        if self._is_server_side and strategy is _cursor._DEFAULT_FETCH:
            strategy = _cursor.BufferedRowCursorFetchStrategy(self.cursor, self.execution_options)
        if strategy is _cursor._NO_CURSOR_DML:
            cursor_description = None
        else:
            cursor_description = strategy.alternate_cursor_description or self.cursor.description
        if cursor_description is None:
            strategy = _cursor._NO_CURSOR_DML
        elif self._num_sentinel_cols:
            assert self.execute_style is ExecuteStyle.INSERTMANYVALUES
            cursor_description = cursor_description[0:-self._num_sentinel_cols]
        result: _cursor.CursorResult[Any] = _cursor.CursorResult(self, strategy, cursor_description)
        if self.isinsert:
            if self._is_implicit_returning:
                rows = result.all()
                self.returned_default_rows = rows
                self.inserted_primary_key_rows = self._setup_ins_pk_from_implicit_returning(result, rows)
                assert result._metadata.returns_rows
                if self._is_supplemental_returning:
                    result._rewind(rows)
                else:
                    result._soft_close()
            elif not self._is_explicit_returning:
                result._soft_close()
        elif self._is_implicit_returning:
            rows = result.all()
            if rows:
                self.returned_default_rows = rows
            self._rowcount = len(rows)
            if self._is_supplemental_returning:
                result._rewind(rows)
            else:
                result._soft_close()
            assert result._metadata.returns_rows
        elif not result._metadata.returns_rows:
            if self._rowcount is None:
                self._rowcount = self.cursor.rowcount
            result._soft_close()
        elif self.isupdate or self.isdelete:
            if self._rowcount is None:
                self._rowcount = self.cursor.rowcount
        return result

    @util.memoized_property
    def inserted_primary_key_rows(self):
        return self._setup_ins_pk_from_empty()

    def _setup_ins_pk_from_lastrowid(self):
        getter = cast(SQLCompiler, self.compiled)._inserted_primary_key_from_lastrowid_getter
        lastrowid = self.get_lastrowid()
        return [getter(lastrowid, self.compiled_parameters[0])]

    def _setup_ins_pk_from_empty(self):
        getter = cast(SQLCompiler, self.compiled)._inserted_primary_key_from_lastrowid_getter
        return [getter(None, param) for param in self.compiled_parameters]

    def _setup_ins_pk_from_implicit_returning(self, result, rows):
        if not rows:
            return []
        getter = cast(SQLCompiler, self.compiled)._inserted_primary_key_from_returning_getter
        compiled_params = self.compiled_parameters
        return [getter(row, param) for row, param in zip(rows, compiled_params)]

    def lastrow_has_defaults(self):
        return (self.isinsert or self.isupdate) and bool(cast(SQLCompiler, self.compiled).postfetch)

    def _prepare_set_input_sizes(self) -> Optional[List[Tuple[str, Any, TypeEngine[Any]]]]:
        """Given a cursor and ClauseParameters, prepare arguments
        in order to call the appropriate
        style of ``setinputsizes()`` on the cursor, using DB-API types
        from the bind parameter's ``TypeEngine`` objects.

        This method only called by those dialects which set
        the :attr:`.Dialect.bind_typing` attribute to
        :attr:`.BindTyping.SETINPUTSIZES`.   cx_Oracle is the only DBAPI
        that requires setinputsizes(), pyodbc offers it as an option.

        Prior to SQLAlchemy 2.0, the setinputsizes() approach was also used
        for pg8000 and asyncpg, which has been changed to inline rendering
        of casts.

        """
        if self.isddl or self.is_text:
            return None
        compiled = cast(SQLCompiler, self.compiled)
        inputsizes = compiled._get_set_input_sizes_lookup()
        if inputsizes is None:
            return None
        dialect = self.dialect
        if dialect._has_events:
            inputsizes = dict(inputsizes)
            dialect.dispatch.do_setinputsizes(inputsizes, self.cursor, self.statement, self.parameters, self)
        if compiled.escaped_bind_names:
            escaped_bind_names = compiled.escaped_bind_names
        else:
            escaped_bind_names = None
        if dialect.positional:
            items = [(key, compiled.binds[key]) for key in compiled.positiontup or ()]
        else:
            items = [(key, bindparam) for bindparam, key in compiled.bind_names.items()]
        generic_inputsizes: List[Tuple[str, Any, TypeEngine[Any]]] = []
        for key, bindparam in items:
            if bindparam in compiled.literal_execute_params:
                continue
            if key in self._expanded_parameters:
                if is_tuple_type(bindparam.type):
                    num = len(bindparam.type.types)
                    dbtypes = inputsizes[bindparam]
                    generic_inputsizes.extend(((escaped_bind_names.get(paramname, paramname) if escaped_bind_names is not None else paramname, dbtypes[idx % num], bindparam.type.types[idx % num]) for idx, paramname in enumerate(self._expanded_parameters[key])))
                else:
                    dbtype = inputsizes.get(bindparam, None)
                    generic_inputsizes.extend(((escaped_bind_names.get(paramname, paramname) if escaped_bind_names is not None else paramname, dbtype, bindparam.type) for paramname in self._expanded_parameters[key]))
            else:
                dbtype = inputsizes.get(bindparam, None)
                escaped_name = escaped_bind_names.get(key, key) if escaped_bind_names is not None else key
                generic_inputsizes.append((escaped_name, dbtype, bindparam.type))
        return generic_inputsizes

    def _exec_default(self, column, default, type_):
        if default.is_sequence:
            return self.fire_sequence(default, type_)
        elif default.is_callable:
            self.current_column = column
            return default.arg(self)
        elif default.is_clause_element:
            return self._exec_default_clause_element(column, default, type_)
        else:
            return default.arg

    def _exec_default_clause_element(self, column, default, type_):
        if not default._arg_is_typed:
            default_arg = expression.type_coerce(default.arg, type_)
        else:
            default_arg = default.arg
        compiled = expression.select(default_arg).compile(dialect=self.dialect)
        compiled_params = compiled.construct_params()
        processors = compiled._bind_processors
        if compiled.positional:
            parameters = self.dialect.execute_sequence_format([processors[key](compiled_params[key]) if key in processors else compiled_params[key] for key in compiled.positiontup or ()])
        else:
            parameters = {key: processors[key](compiled_params[key]) if key in processors else compiled_params[key] for key in compiled_params}
        return self._execute_scalar(str(compiled), type_, parameters=parameters)
    current_parameters: Optional[_CoreSingleExecuteParams] = None
    'A dictionary of parameters applied to the current row.\n\n    This attribute is only available in the context of a user-defined default\n    generation function, e.g. as described at :ref:`context_default_functions`.\n    It consists of a dictionary which includes entries for each column/value\n    pair that is to be part of the INSERT or UPDATE statement. The keys of the\n    dictionary will be the key value of each :class:`_schema.Column`,\n    which is usually\n    synonymous with the name.\n\n    Note that the :attr:`.DefaultExecutionContext.current_parameters` attribute\n    does not accommodate for the "multi-values" feature of the\n    :meth:`_expression.Insert.values` method.  The\n    :meth:`.DefaultExecutionContext.get_current_parameters` method should be\n    preferred.\n\n    .. seealso::\n\n        :meth:`.DefaultExecutionContext.get_current_parameters`\n\n        :ref:`context_default_functions`\n\n    '

    def get_current_parameters(self, isolate_multiinsert_groups=True):
        """Return a dictionary of parameters applied to the current row.

        This method can only be used in the context of a user-defined default
        generation function, e.g. as described at
        :ref:`context_default_functions`. When invoked, a dictionary is
        returned which includes entries for each column/value pair that is part
        of the INSERT or UPDATE statement. The keys of the dictionary will be
        the key value of each :class:`_schema.Column`,
        which is usually synonymous
        with the name.

        :param isolate_multiinsert_groups=True: indicates that multi-valued
         INSERT constructs created using :meth:`_expression.Insert.values`
         should be
         handled by returning only the subset of parameters that are local
         to the current column default invocation.   When ``False``, the
         raw parameters of the statement are returned including the
         naming convention used in the case of multi-valued INSERT.

        .. versionadded:: 1.2  added
           :meth:`.DefaultExecutionContext.get_current_parameters`
           which provides more functionality over the existing
           :attr:`.DefaultExecutionContext.current_parameters`
           attribute.

        .. seealso::

            :attr:`.DefaultExecutionContext.current_parameters`

            :ref:`context_default_functions`

        """
        try:
            parameters = self.current_parameters
            column = self.current_column
        except AttributeError:
            raise exc.InvalidRequestError('get_current_parameters() can only be invoked in the context of a Python side column default function')
        else:
            assert column is not None
            assert parameters is not None
        compile_state = cast('DMLState', cast(SQLCompiler, self.compiled).compile_state)
        assert compile_state is not None
        if isolate_multiinsert_groups and dml.isinsert(compile_state) and compile_state._has_multi_parameters:
            if column._is_multiparam_column:
                index = column.index + 1
                d = {column.original.key: parameters[column.key]}
            else:
                d = {column.key: parameters[column.key]}
                index = 0
            assert compile_state._dict_parameters is not None
            keys = compile_state._dict_parameters.keys()
            d.update(((key, parameters['%s_m%d' % (key, index)]) for key in keys))
            return d
        else:
            return parameters

    def get_insert_default(self, column):
        if column.default is None:
            return None
        else:
            return self._exec_default(column, column.default, column.type)

    def get_update_default(self, column):
        if column.onupdate is None:
            return None
        else:
            return self._exec_default(column, column.onupdate, column.type)

    def _process_execute_defaults(self):
        compiled = cast(SQLCompiler, self.compiled)
        key_getter = compiled._within_exec_param_key_getter
        sentinel_counter = 0
        if compiled.insert_prefetch:
            prefetch_recs = [(c, key_getter(c), c._default_description_tuple, self.get_insert_default) for c in compiled.insert_prefetch]
        elif compiled.update_prefetch:
            prefetch_recs = [(c, key_getter(c), c._onupdate_description_tuple, self.get_update_default) for c in compiled.update_prefetch]
        else:
            prefetch_recs = []
        for param in self.compiled_parameters:
            self.current_parameters = param
            for c, param_key, (arg, is_scalar, is_callable, is_sentinel), fallback in prefetch_recs:
                if is_sentinel:
                    param[param_key] = sentinel_counter
                    sentinel_counter += 1
                elif is_scalar:
                    param[param_key] = arg
                elif is_callable:
                    self.current_column = c
                    param[param_key] = arg(self)
                else:
                    val = fallback(c)
                    if val is not None:
                        param[param_key] = val
        del self.current_parameters