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
class CompileState:
    """Produces additional object state necessary for a statement to be
    compiled.

    the :class:`.CompileState` class is at the base of classes that assemble
    state for a particular statement object that is then used by the
    compiler.   This process is essentially an extension of the process that
    the SQLCompiler.visit_XYZ() method takes, however there is an emphasis
    on converting raw user intent into more organized structures rather than
    producing string output.   The top-level :class:`.CompileState` for the
    statement being executed is also accessible when the execution context
    works with invoking the statement and collecting results.

    The production of :class:`.CompileState` is specific to the compiler,  such
    as within the :meth:`.SQLCompiler.visit_insert`,
    :meth:`.SQLCompiler.visit_select` etc. methods.  These methods are also
    responsible for associating the :class:`.CompileState` with the
    :class:`.SQLCompiler` itself, if the statement is the "toplevel" statement,
    i.e. the outermost SQL statement that's actually being executed.
    There can be other :class:`.CompileState` objects that are not the
    toplevel, such as when a SELECT subquery or CTE-nested
    INSERT/UPDATE/DELETE is generated.

    .. versionadded:: 1.4

    """
    __slots__ = ('statement', '_ambiguous_table_name_map')
    plugins: Dict[Tuple[str, str], Type[CompileState]] = {}
    _ambiguous_table_name_map: Optional[_AmbiguousTableNameMap]

    @classmethod
    def create_for_statement(cls, statement, compiler, **kw):
        if statement._propagate_attrs:
            plugin_name = statement._propagate_attrs.get('compile_state_plugin', 'default')
            klass = cls.plugins.get((plugin_name, statement._effective_plugin_target), None)
            if klass is None:
                klass = cls.plugins['default', statement._effective_plugin_target]
        else:
            klass = cls.plugins['default', statement._effective_plugin_target]
        if klass is cls:
            return cls(statement, compiler, **kw)
        else:
            return klass.create_for_statement(statement, compiler, **kw)

    def __init__(self, statement, compiler, **kw):
        self.statement = statement

    @classmethod
    def get_plugin_class(cls, statement: Executable) -> Optional[Type[CompileState]]:
        plugin_name = statement._propagate_attrs.get('compile_state_plugin', None)
        if plugin_name:
            key = (plugin_name, statement._effective_plugin_target)
            if key in cls.plugins:
                return cls.plugins[key]
        try:
            return cls.plugins['default', statement._effective_plugin_target]
        except KeyError:
            return None

    @classmethod
    def _get_plugin_class_for_plugin(cls, statement: Executable, plugin_name: str) -> Optional[Type[CompileState]]:
        try:
            return cls.plugins[plugin_name, statement._effective_plugin_target]
        except KeyError:
            return None

    @classmethod
    def plugin_for(cls, plugin_name: str, visit_name: str) -> Callable[[_Fn], _Fn]:

        def decorate(cls_to_decorate):
            cls.plugins[plugin_name, visit_name] = cls_to_decorate
            return cls_to_decorate
        return decorate