from __future__ import annotations
import collections
from enum import Enum
import itertools
from itertools import zip_longest
import operator
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import roles
from . import visitors
from .cache_key import HasCacheKey  # noqa
from .cache_key import MemoizedHasCacheKey  # noqa
from .traversals import HasCopyInternals  # noqa
from .visitors import ClauseVisitor
from .visitors import ExtendedInternalTraversal
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import util
from ..util import HasMemoized as HasMemoized
from ..util import hybridmethod
from ..util import typing as compat_typing
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeGuard
class Executable(roles.StatementRole):
    """Mark a :class:`_expression.ClauseElement` as supporting execution.

    :class:`.Executable` is a superclass for all "statement" types
    of objects, including :func:`select`, :func:`delete`, :func:`update`,
    :func:`insert`, :func:`text`.

    """
    supports_execution: bool = True
    _execution_options: _ImmutableExecuteOptions = util.EMPTY_DICT
    _is_default_generator = False
    _with_options: Tuple[ExecutableOption, ...] = ()
    _with_context_options: Tuple[Tuple[Callable[[CompileState], None], Any], ...] = ()
    _compile_options: Optional[Union[Type[CacheableOptions], CacheableOptions]]
    _executable_traverse_internals = [('_with_options', InternalTraversal.dp_executable_options), ('_with_context_options', ExtendedInternalTraversal.dp_with_context_options), ('_propagate_attrs', ExtendedInternalTraversal.dp_propagate_attrs)]
    is_select = False
    is_update = False
    is_insert = False
    is_text = False
    is_delete = False
    is_dml = False
    if TYPE_CHECKING:
        __visit_name__: str

        def _compile_w_cache(self, dialect: Dialect, *, compiled_cache: Optional[CompiledCacheType], column_keys: List[str], for_executemany: bool=False, schema_translate_map: Optional[SchemaTranslateMapType]=None, **kw: Any) -> Tuple[Compiled, Optional[Sequence[BindParameter[Any]]], CacheStats]:
            ...

        def _execute_on_connection(self, connection: Connection, distilled_params: _CoreMultiExecuteParams, execution_options: CoreExecuteOptionsParameter) -> CursorResult[Any]:
            ...

        def _execute_on_scalar(self, connection: Connection, distilled_params: _CoreMultiExecuteParams, execution_options: CoreExecuteOptionsParameter) -> Any:
            ...

    @util.ro_non_memoized_property
    def _all_selected_columns(self):
        raise NotImplementedError()

    @property
    def _effective_plugin_target(self) -> str:
        return self.__visit_name__

    @_generative
    def options(self, *options: ExecutableOption) -> Self:
        """Apply options to this statement.

        In the general sense, options are any kind of Python object
        that can be interpreted by the SQL compiler for the statement.
        These options can be consumed by specific dialects or specific kinds
        of compilers.

        The most commonly known kind of option are the ORM level options
        that apply "eager load" and other loading behaviors to an ORM
        query.   However, options can theoretically be used for many other
        purposes.

        For background on specific kinds of options for specific kinds of
        statements, refer to the documentation for those option objects.

        .. versionchanged:: 1.4 - added :meth:`.Executable.options` to
           Core statement objects towards the goal of allowing unified
           Core / ORM querying capabilities.

        .. seealso::

            :ref:`loading_columns` - refers to options specific to the usage
            of ORM queries

            :ref:`relationship_loader_options` - refers to options specific
            to the usage of ORM queries

        """
        self._with_options += tuple((coercions.expect(roles.ExecutableOptionRole, opt) for opt in options))
        return self

    @_generative
    def _set_compile_options(self, compile_options: CacheableOptions) -> Self:
        """Assign the compile options to a new value.

        :param compile_options: appropriate CacheableOptions structure

        """
        self._compile_options = compile_options
        return self

    @_generative
    def _update_compile_options(self, options: CacheableOptions) -> Self:
        """update the _compile_options with new keys."""
        assert self._compile_options is not None
        self._compile_options += options
        return self

    @_generative
    def _add_context_option(self, callable_: Callable[[CompileState], None], cache_args: Any) -> Self:
        """Add a context option to this statement.

        These are callable functions that will
        be given the CompileState object upon compilation.

        A second argument cache_args is required, which will be combined with
        the ``__code__`` identity of the function itself in order to produce a
        cache key.

        """
        self._with_context_options += ((callable_, cache_args),)
        return self

    @overload
    def execution_options(self, *, compiled_cache: Optional[CompiledCacheType]=..., logging_token: str=..., isolation_level: IsolationLevel=..., no_parameters: bool=False, stream_results: bool=False, max_row_buffer: int=..., yield_per: int=..., insertmanyvalues_page_size: int=..., schema_translate_map: Optional[SchemaTranslateMapType]=..., populate_existing: bool=False, autoflush: bool=False, synchronize_session: SynchronizeSessionArgument=..., dml_strategy: DMLStrategyArgument=..., render_nulls: bool=..., is_delete_using: bool=..., is_update_from: bool=..., preserve_rowcount: bool=False, **opt: Any) -> Self:
        ...

    @overload
    def execution_options(self, **opt: Any) -> Self:
        ...

    @_generative
    def execution_options(self, **kw: Any) -> Self:
        """Set non-SQL options for the statement which take effect during
        execution.

        Execution options can be set at many scopes, including per-statement,
        per-connection, or per execution, using methods such as
        :meth:`_engine.Connection.execution_options` and parameters which
        accept a dictionary of options such as
        :paramref:`_engine.Connection.execute.execution_options` and
        :paramref:`_orm.Session.execute.execution_options`.

        The primary characteristic of an execution option, as opposed to
        other kinds of options such as ORM loader options, is that
        **execution options never affect the compiled SQL of a query, only
        things that affect how the SQL statement itself is invoked or how
        results are fetched**.  That is, execution options are not part of
        what's accommodated by SQL compilation nor are they considered part of
        the cached state of a statement.

        The :meth:`_sql.Executable.execution_options` method is
        :term:`generative`, as
        is the case for the method as applied to the :class:`_engine.Engine`
        and :class:`_orm.Query` objects, which means when the method is called,
        a copy of the object is returned, which applies the given parameters to
        that new copy, but leaves the original unchanged::

            statement = select(table.c.x, table.c.y)
            new_statement = statement.execution_options(my_option=True)

        An exception to this behavior is the :class:`_engine.Connection`
        object, where the :meth:`_engine.Connection.execution_options` method
        is explicitly **not** generative.

        The kinds of options that may be passed to
        :meth:`_sql.Executable.execution_options` and other related methods and
        parameter dictionaries include parameters that are explicitly consumed
        by SQLAlchemy Core or ORM, as well as arbitrary keyword arguments not
        defined by SQLAlchemy, which means the methods and/or parameter
        dictionaries may be used for user-defined parameters that interact with
        custom code, which may access the parameters using methods such as
        :meth:`_sql.Executable.get_execution_options` and
        :meth:`_engine.Connection.get_execution_options`, or within selected
        event hooks using a dedicated ``execution_options`` event parameter
        such as
        :paramref:`_events.ConnectionEvents.before_execute.execution_options`
        or :attr:`_orm.ORMExecuteState.execution_options`, e.g.::

             from sqlalchemy import event

             @event.listens_for(some_engine, "before_execute")
             def _process_opt(conn, statement, multiparams, params, execution_options):
                 "run a SQL function before invoking a statement"

                 if execution_options.get("do_special_thing", False):
                     conn.exec_driver_sql("run_special_function()")

        Within the scope of options that are explicitly recognized by
        SQLAlchemy, most apply to specific classes of objects and not others.
        The most common execution options include:

        * :paramref:`_engine.Connection.execution_options.isolation_level` -
          sets the isolation level for a connection or a class of connections
          via an :class:`_engine.Engine`.  This option is accepted only
          by :class:`_engine.Connection` or :class:`_engine.Engine`.

        * :paramref:`_engine.Connection.execution_options.stream_results` -
          indicates results should be fetched using a server side cursor;
          this option is accepted by :class:`_engine.Connection`, by the
          :paramref:`_engine.Connection.execute.execution_options` parameter
          on :meth:`_engine.Connection.execute`, and additionally by
          :meth:`_sql.Executable.execution_options` on a SQL statement object,
          as well as by ORM constructs like :meth:`_orm.Session.execute`.

        * :paramref:`_engine.Connection.execution_options.compiled_cache` -
          indicates a dictionary that will serve as the
          :ref:`SQL compilation cache <sql_caching>`
          for a :class:`_engine.Connection` or :class:`_engine.Engine`, as
          well as for ORM methods like :meth:`_orm.Session.execute`.
          Can be passed as ``None`` to disable caching for statements.
          This option is not accepted by
          :meth:`_sql.Executable.execution_options` as it is inadvisable to
          carry along a compilation cache within a statement object.

        * :paramref:`_engine.Connection.execution_options.schema_translate_map`
          - a mapping of schema names used by the
          :ref:`Schema Translate Map <schema_translating>` feature, accepted
          by :class:`_engine.Connection`, :class:`_engine.Engine`,
          :class:`_sql.Executable`, as well as by ORM constructs
          like :meth:`_orm.Session.execute`.

        .. seealso::

            :meth:`_engine.Connection.execution_options`

            :paramref:`_engine.Connection.execute.execution_options`

            :paramref:`_orm.Session.execute.execution_options`

            :ref:`orm_queryguide_execution_options` - documentation on all
            ORM-specific execution options

        """
        if 'isolation_level' in kw:
            raise exc.ArgumentError("'isolation_level' execution option may only be specified on Connection.execution_options(), or per-engine using the isolation_level argument to create_engine().")
        if 'compiled_cache' in kw:
            raise exc.ArgumentError("'compiled_cache' execution option may only be specified on Connection.execution_options(), not per statement.")
        self._execution_options = self._execution_options.union(kw)
        return self

    def get_execution_options(self) -> _ExecuteOptions:
        """Get the non-SQL options which will take effect during execution.

        .. versionadded:: 1.3

        .. seealso::

            :meth:`.Executable.execution_options`
        """
        return self._execution_options