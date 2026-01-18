from __future__ import annotations
from collections import deque
from enum import Enum
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util import langhelpers
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
class HasTraversalDispatch:
    """Define infrastructure for classes that perform internal traversals

    .. versionadded:: 2.0

    """
    __slots__ = ()
    _dispatch_lookup: ClassVar[Dict[Union[InternalTraversal, str], str]] = {}

    def dispatch(self, visit_symbol: InternalTraversal) -> Callable[..., Any]:
        """Given a method from :class:`.HasTraversalDispatch`, return the
        corresponding method on a subclass.

        """
        name = _dispatch_lookup[visit_symbol]
        return getattr(self, name, None)

    def run_generated_dispatch(self, target: object, internal_dispatch: _TraverseInternalsType, generate_dispatcher_name: str) -> Any:
        dispatcher: _InternalTraversalDispatchType
        try:
            dispatcher = target.__class__.__dict__[generate_dispatcher_name]
        except KeyError:
            dispatcher = self.generate_dispatch(target.__class__, internal_dispatch, generate_dispatcher_name)
        return dispatcher(target, self)

    def generate_dispatch(self, target_cls: Type[object], internal_dispatch: _TraverseInternalsType, generate_dispatcher_name: str) -> _InternalTraversalDispatchType:
        dispatcher = self._generate_dispatcher(internal_dispatch, generate_dispatcher_name)
        setattr(target_cls, generate_dispatcher_name, dispatcher)
        return dispatcher

    def _generate_dispatcher(self, internal_dispatch: _TraverseInternalsType, method_name: str) -> _InternalTraversalDispatchType:
        names = []
        for attrname, visit_sym in internal_dispatch:
            meth = self.dispatch(visit_sym)
            if meth is not None:
                visit_name = _dispatch_lookup[visit_sym]
                names.append((attrname, visit_name))
        code = '    return [\n' + ', \n'.join(('        (%r, self.%s, visitor.%s)' % (attrname, attrname, visit_name) for attrname, visit_name in names)) + '\n    ]\n'
        meth_text = 'def %s(self, visitor):\n' % method_name + code + '\n'
        return cast(_InternalTraversalDispatchType, langhelpers._exec_code_in_env(meth_text, {}, method_name))