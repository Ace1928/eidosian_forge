from __future__ import annotations
import contextlib
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence as typing_Sequence
from typing import Tuple
from . import roles
from .base import _generative
from .base import Executable
from .base import SchemaVisitor
from .elements import ClauseElement
from .. import exc
from .. import util
from ..util import topological
from ..util.typing import Protocol
from ..util.typing import Self
class ExecutableDDLElement(roles.DDLRole, Executable, BaseDDLElement):
    """Base class for standalone executable DDL expression constructs.

    This class is the base for the general purpose :class:`.DDL` class,
    as well as the various create/drop clause constructs such as
    :class:`.CreateTable`, :class:`.DropTable`, :class:`.AddConstraint`,
    etc.

    .. versionchanged:: 2.0  :class:`.ExecutableDDLElement` is renamed from
       :class:`.DDLElement`, which still exists for backwards compatibility.

    :class:`.ExecutableDDLElement` integrates closely with SQLAlchemy events,
    introduced in :ref:`event_toplevel`.  An instance of one is
    itself an event receiving callable::

        event.listen(
            users,
            'after_create',
            AddConstraint(constraint).execute_if(dialect='postgresql')
        )

    .. seealso::

        :class:`.DDL`

        :class:`.DDLEvents`

        :ref:`event_toplevel`

        :ref:`schema_ddl_sequences`

    """
    _ddl_if: Optional[DDLIf] = None
    target: Optional[SchemaItem] = None

    def _execute_on_connection(self, connection, distilled_params, execution_options):
        return connection._execute_ddl(self, distilled_params, execution_options)

    @_generative
    def against(self, target: SchemaItem) -> Self:
        """Return a copy of this :class:`_schema.ExecutableDDLElement` which
        will include the given target.

        This essentially applies the given item to the ``.target`` attribute of
        the returned :class:`_schema.ExecutableDDLElement` object. This target
        is then usable by event handlers and compilation routines in order to
        provide services such as tokenization of a DDL string in terms of a
        particular :class:`_schema.Table`.

        When a :class:`_schema.ExecutableDDLElement` object is established as
        an event handler for the :meth:`_events.DDLEvents.before_create` or
        :meth:`_events.DDLEvents.after_create` events, and the event then
        occurs for a given target such as a :class:`_schema.Constraint` or
        :class:`_schema.Table`, that target is established with a copy of the
        :class:`_schema.ExecutableDDLElement` object using this method, which
        then proceeds to the :meth:`_schema.ExecutableDDLElement.execute`
        method in order to invoke the actual DDL instruction.

        :param target: a :class:`_schema.SchemaItem` that will be the subject
         of a DDL operation.

        :return: a copy of this :class:`_schema.ExecutableDDLElement` with the
         ``.target`` attribute assigned to the given
         :class:`_schema.SchemaItem`.

        .. seealso::

            :class:`_schema.DDL` - uses tokenization against the "target" when
            processing the DDL string.

        """
        self.target = target
        return self

    @_generative
    def execute_if(self, dialect: Optional[str]=None, callable_: Optional[DDLIfCallable]=None, state: Optional[Any]=None) -> Self:
        """Return a callable that will execute this
        :class:`_ddl.ExecutableDDLElement` conditionally within an event
        handler.

        Used to provide a wrapper for event listening::

            event.listen(
                        metadata,
                        'before_create',
                        DDL("my_ddl").execute_if(dialect='postgresql')
                    )

        :param dialect: May be a string or tuple of strings.
          If a string, it will be compared to the name of the
          executing database dialect::

            DDL('something').execute_if(dialect='postgresql')

          If a tuple, specifies multiple dialect names::

            DDL('something').execute_if(dialect=('postgresql', 'mysql'))

        :param callable\\_: A callable, which will be invoked with
          three positional arguments as well as optional keyword
          arguments:

            :ddl:
              This DDL element.

            :target:
              The :class:`_schema.Table` or :class:`_schema.MetaData`
              object which is the
              target of this event. May be None if the DDL is executed
              explicitly.

            :bind:
              The :class:`_engine.Connection` being used for DDL execution.
              May be None if this construct is being created inline within
              a table, in which case ``compiler`` will be present.

            :tables:
              Optional keyword argument - a list of Table objects which are to
              be created/ dropped within a MetaData.create_all() or drop_all()
              method call.

            :dialect: keyword argument, but always present - the
              :class:`.Dialect` involved in the operation.

            :compiler: keyword argument.  Will be ``None`` for an engine
              level DDL invocation, but will refer to a :class:`.DDLCompiler`
              if this DDL element is being created inline within a table.

            :state:
              Optional keyword argument - will be the ``state`` argument
              passed to this function.

            :checkfirst:
             Keyword argument, will be True if the 'checkfirst' flag was
             set during the call to ``create()``, ``create_all()``,
             ``drop()``, ``drop_all()``.

          If the callable returns a True value, the DDL statement will be
          executed.

        :param state: any value which will be passed to the callable\\_
          as the ``state`` keyword argument.

        .. seealso::

            :meth:`.SchemaItem.ddl_if`

            :class:`.DDLEvents`

            :ref:`event_toplevel`

        """
        self._ddl_if = DDLIf(dialect, callable_, state)
        return self

    def _should_execute(self, target, bind, **kw):
        if self._ddl_if is None:
            return True
        else:
            return self._ddl_if._should_execute(self, target, bind, **kw)

    def _invoke_with(self, bind):
        if self._should_execute(self.target, bind):
            return bind.execute(self)

    def __call__(self, target, bind, **kw):
        """Execute the DDL as a ddl_listener."""
        self.against(target)._invoke_with(bind)

    def _generate(self):
        s = self.__class__.__new__(self.__class__)
        s.__dict__ = self.__dict__.copy()
        return s