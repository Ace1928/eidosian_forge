from __future__ import annotations
import contextlib
from enum import Enum
import itertools
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
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
import weakref
from . import attributes
from . import bulk_persistence
from . import context
from . import descriptor_props
from . import exc
from . import identity
from . import loading
from . import query
from . import state as statelib
from ._typing import _O
from ._typing import insp_is_mapper
from ._typing import is_composite_class
from ._typing import is_orm_option
from ._typing import is_user_defined_option
from .base import _class_to_mapper
from .base import _none_set
from .base import _state_mapper
from .base import instance_str
from .base import LoaderCallableStatus
from .base import object_mapper
from .base import object_state
from .base import PassiveFlag
from .base import state_str
from .context import FromStatement
from .context import ORMCompileState
from .identity import IdentityMap
from .query import Query
from .state import InstanceState
from .state_changes import _StateChange
from .state_changes import _StateChangeState
from .state_changes import _StateChangeStates
from .unitofwork import UOWTransaction
from .. import engine
from .. import exc as sa_exc
from .. import sql
from .. import util
from ..engine import Connection
from ..engine import Engine
from ..engine.util import TransactionalContext
from ..event import dispatcher
from ..event import EventTarget
from ..inspection import inspect
from ..inspection import Inspectable
from ..sql import coercions
from ..sql import dml
from ..sql import roles
from ..sql import Select
from ..sql import TableClause
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import CompileState
from ..sql.schema import Table
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import IdentitySet
from ..util.typing import Literal
from ..util.typing import Protocol
class ORMExecuteState(util.MemoizedSlots):
    """Represents a call to the :meth:`_orm.Session.execute` method, as passed
    to the :meth:`.SessionEvents.do_orm_execute` event hook.

    .. versionadded:: 1.4

    .. seealso::

        :ref:`session_execute_events` - top level documentation on how
        to use :meth:`_orm.SessionEvents.do_orm_execute`

    """
    __slots__ = ('session', 'statement', 'parameters', 'execution_options', 'local_execution_options', 'bind_arguments', 'identity_token', '_compile_state_cls', '_starting_event_idx', '_events_todo', '_update_execution_options')
    session: Session
    'The :class:`_orm.Session` in use.'
    statement: Executable
    'The SQL statement being invoked.\n\n    For an ORM selection as would\n    be retrieved from :class:`_orm.Query`, this is an instance of\n    :class:`_sql.select` that was generated from the ORM query.\n    '
    parameters: Optional[_CoreAnyExecuteParams]
    'Dictionary of parameters that was passed to\n    :meth:`_orm.Session.execute`.'
    execution_options: _ExecuteOptions
    'The complete dictionary of current execution options.\n\n    This is a merge of the statement level options with the\n    locally passed execution options.\n\n    .. seealso::\n\n        :attr:`_orm.ORMExecuteState.local_execution_options`\n\n        :meth:`_sql.Executable.execution_options`\n\n        :ref:`orm_queryguide_execution_options`\n\n    '
    local_execution_options: _ExecuteOptions
    'Dictionary view of the execution options passed to the\n    :meth:`.Session.execute` method.\n\n    This does not include options that may be associated with the statement\n    being invoked.\n\n    .. seealso::\n\n        :attr:`_orm.ORMExecuteState.execution_options`\n\n    '
    bind_arguments: _BindArguments
    'The dictionary passed as the\n    :paramref:`_orm.Session.execute.bind_arguments` dictionary.\n\n    This dictionary may be used by extensions to :class:`_orm.Session` to pass\n    arguments that will assist in determining amongst a set of database\n    connections which one should be used to invoke this statement.\n\n    '
    _compile_state_cls: Optional[Type[ORMCompileState]]
    _starting_event_idx: int
    _events_todo: List[Any]
    _update_execution_options: Optional[_ExecuteOptions]

    def __init__(self, session: Session, statement: Executable, parameters: Optional[_CoreAnyExecuteParams], execution_options: _ExecuteOptions, bind_arguments: _BindArguments, compile_state_cls: Optional[Type[ORMCompileState]], events_todo: List[_InstanceLevelDispatch[Session]]):
        """Construct a new :class:`_orm.ORMExecuteState`.

        this object is constructed internally.

        """
        self.session = session
        self.statement = statement
        self.parameters = parameters
        self.local_execution_options = execution_options
        self.execution_options = statement._execution_options.union(execution_options)
        self.bind_arguments = bind_arguments
        self._compile_state_cls = compile_state_cls
        self._events_todo = list(events_todo)

    def _remaining_events(self) -> List[_InstanceLevelDispatch[Session]]:
        return self._events_todo[self._starting_event_idx + 1:]

    def invoke_statement(self, statement: Optional[Executable]=None, params: Optional[_CoreAnyExecuteParams]=None, execution_options: Optional[OrmExecuteOptionsParameter]=None, bind_arguments: Optional[_BindArguments]=None) -> Result[Any]:
        """Execute the statement represented by this
        :class:`.ORMExecuteState`, without re-invoking events that have
        already proceeded.

        This method essentially performs a re-entrant execution of the current
        statement for which the :meth:`.SessionEvents.do_orm_execute` event is
        being currently invoked.    The use case for this is for event handlers
        that want to override how the ultimate
        :class:`_engine.Result` object is returned, such as for schemes that
        retrieve results from an offline cache or which concatenate results
        from multiple executions.

        When the :class:`_engine.Result` object is returned by the actual
        handler function within :meth:`_orm.SessionEvents.do_orm_execute` and
        is propagated to the calling
        :meth:`_orm.Session.execute` method, the remainder of the
        :meth:`_orm.Session.execute` method is preempted and the
        :class:`_engine.Result` object is returned to the caller of
        :meth:`_orm.Session.execute` immediately.

        :param statement: optional statement to be invoked, in place of the
         statement currently represented by :attr:`.ORMExecuteState.statement`.

        :param params: optional dictionary of parameters or list of parameters
         which will be merged into the existing
         :attr:`.ORMExecuteState.parameters` of this :class:`.ORMExecuteState`.

         .. versionchanged:: 2.0 a list of parameter dictionaries is accepted
            for executemany executions.

        :param execution_options: optional dictionary of execution options
         will be merged into the existing
         :attr:`.ORMExecuteState.execution_options` of this
         :class:`.ORMExecuteState`.

        :param bind_arguments: optional dictionary of bind_arguments
         which will be merged amongst the current
         :attr:`.ORMExecuteState.bind_arguments`
         of this :class:`.ORMExecuteState`.

        :return: a :class:`_engine.Result` object with ORM-level results.

        .. seealso::

            :ref:`do_orm_execute_re_executing` - background and examples on the
            appropriate usage of :meth:`_orm.ORMExecuteState.invoke_statement`.


        """
        if statement is None:
            statement = self.statement
        _bind_arguments = dict(self.bind_arguments)
        if bind_arguments:
            _bind_arguments.update(bind_arguments)
        _bind_arguments['_sa_skip_events'] = True
        _params: Optional[_CoreAnyExecuteParams]
        if params:
            if self.is_executemany:
                _params = []
                exec_many_parameters = cast('List[Dict[str, Any]]', self.parameters)
                for _existing_params, _new_params in itertools.zip_longest(exec_many_parameters, cast('List[Dict[str, Any]]', params)):
                    if _existing_params is None or _new_params is None:
                        raise sa_exc.InvalidRequestError(f"Can't apply executemany parameters to statement; number of parameter sets passed to Session.execute() ({len(exec_many_parameters)}) does not match number of parameter sets given to ORMExecuteState.invoke_statement() ({len(params)})")
                    _existing_params = dict(_existing_params)
                    _existing_params.update(_new_params)
                    _params.append(_existing_params)
            else:
                _params = dict(cast('Dict[str, Any]', self.parameters))
                _params.update(cast('Dict[str, Any]', params))
        else:
            _params = self.parameters
        _execution_options = self.local_execution_options
        if execution_options:
            _execution_options = _execution_options.union(execution_options)
        return self.session._execute_internal(statement, _params, execution_options=_execution_options, bind_arguments=_bind_arguments, _parent_execute_state=self)

    @property
    def bind_mapper(self) -> Optional[Mapper[Any]]:
        """Return the :class:`_orm.Mapper` that is the primary "bind" mapper.

        For an :class:`_orm.ORMExecuteState` object invoking an ORM
        statement, that is, the :attr:`_orm.ORMExecuteState.is_orm_statement`
        attribute is ``True``, this attribute will return the
        :class:`_orm.Mapper` that is considered to be the "primary" mapper
        of the statement.   The term "bind mapper" refers to the fact that
        a :class:`_orm.Session` object may be "bound" to multiple
        :class:`_engine.Engine` objects keyed to mapped classes, and the
        "bind mapper" determines which of those :class:`_engine.Engine` objects
        would be selected.

        For a statement that is invoked against a single mapped class,
        :attr:`_orm.ORMExecuteState.bind_mapper` is intended to be a reliable
        way of getting this mapper.

        .. versionadded:: 1.4.0b2

        .. seealso::

            :attr:`_orm.ORMExecuteState.all_mappers`


        """
        mp: Optional[Mapper[Any]] = self.bind_arguments.get('mapper', None)
        return mp

    @property
    def all_mappers(self) -> Sequence[Mapper[Any]]:
        """Return a sequence of all :class:`_orm.Mapper` objects that are
        involved at the top level of this statement.

        By "top level" we mean those :class:`_orm.Mapper` objects that would
        be represented in the result set rows for a :func:`_sql.select`
        query, or for a :func:`_dml.update` or :func:`_dml.delete` query,
        the mapper that is the main subject of the UPDATE or DELETE.

        .. versionadded:: 1.4.0b2

        .. seealso::

            :attr:`_orm.ORMExecuteState.bind_mapper`



        """
        if not self.is_orm_statement:
            return []
        elif isinstance(self.statement, (Select, FromStatement)):
            result = []
            seen = set()
            for d in self.statement.column_descriptions:
                ent = d['entity']
                if ent:
                    insp = inspect(ent, raiseerr=False)
                    if insp and insp.mapper and (insp.mapper not in seen):
                        seen.add(insp.mapper)
                        result.append(insp.mapper)
            return result
        elif self.statement.is_dml and self.bind_mapper:
            return [self.bind_mapper]
        else:
            return []

    @property
    def is_orm_statement(self) -> bool:
        """return True if the operation is an ORM statement.

        This indicates that the select(), insert(), update(), or delete()
        being invoked contains ORM entities as subjects.   For a statement
        that does not have ORM entities and instead refers only to
        :class:`.Table` metadata, it is invoked as a Core SQL statement
        and no ORM-level automation takes place.

        """
        return self._compile_state_cls is not None

    @property
    def is_executemany(self) -> bool:
        """return True if the parameters are a multi-element list of
        dictionaries with more than one dictionary.

        .. versionadded:: 2.0

        """
        return isinstance(self.parameters, list)

    @property
    def is_select(self) -> bool:
        """return True if this is a SELECT operation."""
        return self.statement.is_select

    @property
    def is_insert(self) -> bool:
        """return True if this is an INSERT operation."""
        return self.statement.is_dml and self.statement.is_insert

    @property
    def is_update(self) -> bool:
        """return True if this is an UPDATE operation."""
        return self.statement.is_dml and self.statement.is_update

    @property
    def is_delete(self) -> bool:
        """return True if this is a DELETE operation."""
        return self.statement.is_dml and self.statement.is_delete

    @property
    def _is_crud(self) -> bool:
        return isinstance(self.statement, (dml.Update, dml.Delete))

    def update_execution_options(self, **opts: Any) -> None:
        """Update the local execution options with new values."""
        self.local_execution_options = self.local_execution_options.union(opts)

    def _orm_compile_options(self) -> Optional[Union[context.ORMCompileState.default_compile_options, Type[context.ORMCompileState.default_compile_options]]]:
        if not self.is_select:
            return None
        try:
            opts = self.statement._compile_options
        except AttributeError:
            return None
        if opts is not None and opts.isinstance(context.ORMCompileState.default_compile_options):
            return opts
        else:
            return None

    @property
    def lazy_loaded_from(self) -> Optional[InstanceState[Any]]:
        """An :class:`.InstanceState` that is using this statement execution
        for a lazy load operation.

        The primary rationale for this attribute is to support the horizontal
        sharding extension, where it is available within specific query
        execution time hooks created by this extension.   To that end, the
        attribute is only intended to be meaningful at **query execution
        time**, and importantly not any time prior to that, including query
        compilation time.

        """
        return self.load_options._lazy_loaded_from

    @property
    def loader_strategy_path(self) -> Optional[PathRegistry]:
        """Return the :class:`.PathRegistry` for the current load path.

        This object represents the "path" in a query along relationships
        when a particular object or collection is being loaded.

        """
        opts = self._orm_compile_options()
        if opts is not None:
            return opts._current_path
        else:
            return None

    @property
    def is_column_load(self) -> bool:
        """Return True if the operation is refreshing column-oriented
        attributes on an existing ORM object.

        This occurs during operations such as :meth:`_orm.Session.refresh`,
        as well as when an attribute deferred by :func:`_orm.defer` is
        being loaded, or an attribute that was expired either directly
        by :meth:`_orm.Session.expire` or via a commit operation is being
        loaded.

        Handlers will very likely not want to add any options to queries
        when such an operation is occurring as the query should be a straight
        primary key fetch which should not have any additional WHERE criteria,
        and loader options travelling with the instance
        will have already been added to the query.

        .. versionadded:: 1.4.0b2

        .. seealso::

            :attr:`_orm.ORMExecuteState.is_relationship_load`

        """
        opts = self._orm_compile_options()
        return opts is not None and opts._for_refresh_state

    @property
    def is_relationship_load(self) -> bool:
        """Return True if this load is loading objects on behalf of a
        relationship.

        This means, the loader in effect is either a LazyLoader,
        SelectInLoader, SubqueryLoader, or similar, and the entire
        SELECT statement being emitted is on behalf of a relationship
        load.

        Handlers will very likely not want to add any options to queries
        when such an operation is occurring, as loader options are already
        capable of being propagated to relationship loaders and should
        be already present.

        .. seealso::

            :attr:`_orm.ORMExecuteState.is_column_load`

        """
        opts = self._orm_compile_options()
        if opts is None:
            return False
        path = self.loader_strategy_path
        return path is not None and (not path.is_root)

    @property
    def load_options(self) -> Union[context.QueryContext.default_load_options, Type[context.QueryContext.default_load_options]]:
        """Return the load_options that will be used for this execution."""
        if not self.is_select:
            raise sa_exc.InvalidRequestError('This ORM execution is not against a SELECT statement so there are no load options.')
        lo: Union[context.QueryContext.default_load_options, Type[context.QueryContext.default_load_options]] = self.execution_options.get('_sa_orm_load_options', context.QueryContext.default_load_options)
        return lo

    @property
    def update_delete_options(self) -> Union[bulk_persistence.BulkUDCompileState.default_update_options, Type[bulk_persistence.BulkUDCompileState.default_update_options]]:
        """Return the update_delete_options that will be used for this
        execution."""
        if not self._is_crud:
            raise sa_exc.InvalidRequestError('This ORM execution is not against an UPDATE or DELETE statement so there are no update options.')
        uo: Union[bulk_persistence.BulkUDCompileState.default_update_options, Type[bulk_persistence.BulkUDCompileState.default_update_options]] = self.execution_options.get('_sa_orm_update_options', bulk_persistence.BulkUDCompileState.default_update_options)
        return uo

    @property
    def _non_compile_orm_options(self) -> Sequence[ORMOption]:
        return [opt for opt in self.statement._with_options if is_orm_option(opt) and (not opt._is_compile_state)]

    @property
    def user_defined_options(self) -> Sequence[UserDefinedOption]:
        """The sequence of :class:`.UserDefinedOptions` that have been
        associated with the statement being invoked.

        """
        return [opt for opt in self.statement._with_options if is_user_defined_option(opt)]