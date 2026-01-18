from __future__ import annotations
from abc import ABC
import collections
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence as _typing_Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import ddl
from . import roles
from . import type_api
from . import visitors
from .base import _DefaultDescriptionTuple
from .base import _NoneName
from .base import _SentinelColumnCharacterization
from .base import _SentinelDefaultCharacterization
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import SchemaEventTarget as SchemaEventTarget
from .coercions import _document_text_coercion
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import quoted_name
from .elements import TextClause
from .selectable import TableClause
from .type_api import to_instance
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
def _maybe_wrap_callable(self, fn: Union[_CallableColumnDefaultProtocol, Callable[[], Any]]) -> _CallableColumnDefaultProtocol:
    """Wrap callables that don't accept a context.

        This is to allow easy compatibility with default callables
        that aren't specific to accepting of a context.

        """
    try:
        argspec = util.get_callable_argspec(fn, no_self=True)
    except TypeError:
        return util.wrap_callable(lambda ctx: fn(), fn)
    defaulted = argspec[3] is not None and len(argspec[3]) or 0
    positionals = len(argspec[0]) - defaulted
    if positionals == 0:
        return util.wrap_callable(lambda ctx: fn(), fn)
    elif positionals == 1:
        return fn
    else:
        raise exc.ArgumentError('ColumnDefault Python function takes zero or one positional arguments')