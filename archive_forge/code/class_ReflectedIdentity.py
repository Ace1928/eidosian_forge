from __future__ import annotations
from enum import Enum
from types import ModuleType
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..event import EventTarget
from ..pool import Pool
from ..pool import PoolProxiedConnection
from ..sql.compiler import Compiled as Compiled
from ..sql.compiler import Compiled  # noqa
from ..sql.compiler import TypeCompiler as TypeCompiler
from ..sql.compiler import TypeCompiler  # noqa
from ..util import immutabledict
from ..util.concurrency import await_only
from ..util.typing import Literal
from ..util.typing import NotRequired
from ..util.typing import Protocol
from ..util.typing import TypedDict
class ReflectedIdentity(TypedDict):
    """represent the reflected IDENTITY structure of a column, corresponding
    to the :class:`_schema.Identity` construct.

    The :class:`.ReflectedIdentity` structure is part of the
    :class:`.ReflectedColumn` structure, which is returned by the
    :meth:`.Inspector.get_columns` method.

    """
    always: bool
    'type of identity column'
    on_null: bool
    'indicates ON NULL'
    start: int
    'starting index of the sequence'
    increment: int
    'increment value of the sequence'
    minvalue: int
    'the minimum value of the sequence.'
    maxvalue: int
    'the maximum value of the sequence.'
    nominvalue: bool
    'no minimum value of the sequence.'
    nomaxvalue: bool
    'no maximum value of the sequence.'
    cycle: bool
    'allows the sequence to wrap around when the maxvalue\n    or minvalue has been reached.'
    cache: Optional[int]
    'number of future values in the\n    sequence which are calculated in advance.'
    order: bool
    'if true, renders the ORDER keyword.'