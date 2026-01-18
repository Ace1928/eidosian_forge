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
class Visitable:
    """Base class for visitable objects.

    :class:`.Visitable` is used to implement the SQL compiler dispatch
    functions.    Other forms of traversal such as for cache key generation
    are implemented separately using the :class:`.HasTraverseInternals`
    interface.

    .. versionchanged:: 2.0  The :class:`.Visitable` class was named
       :class:`.Traversible` in the 1.4 series; the name is changed back
       to :class:`.Visitable` in 2.0 which is what it was prior to 1.4.

       Both names remain importable in both 1.4 and 2.0 versions.

    """
    __slots__ = ()
    __visit_name__: str
    _original_compiler_dispatch: _CompilerDispatchType
    if typing.TYPE_CHECKING:

        def _compiler_dispatch(self, visitor: Any, **kw: Any) -> str:
            ...

    def __init_subclass__(cls) -> None:
        if '__visit_name__' in cls.__dict__:
            cls._generate_compiler_dispatch()
        super().__init_subclass__()

    @classmethod
    def _generate_compiler_dispatch(cls) -> None:
        visit_name = cls.__visit_name__
        if '_compiler_dispatch' in cls.__dict__:
            cls._original_compiler_dispatch = cls._compiler_dispatch
            return
        if not isinstance(visit_name, str):
            raise exc.InvalidRequestError(f'__visit_name__ on class {cls.__name__} must be a string at the class level')
        name = 'visit_%s' % visit_name
        getter = operator.attrgetter(name)

        def _compiler_dispatch(self: Visitable, visitor: Any, **kw: Any) -> str:
            """Look for an attribute named "visit_<visit_name>" on the
            visitor, and call it with the same kw params.

            """
            try:
                meth = getter(visitor)
            except AttributeError as err:
                return visitor.visit_unsupported_compilation(self, err, **kw)
            else:
                return meth(self, **kw)
        cls._compiler_dispatch = cls._original_compiler_dispatch = _compiler_dispatch

    def __class_getitem__(cls, key: Any) -> Any:
        return cls