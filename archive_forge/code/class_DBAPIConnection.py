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
class DBAPIConnection(Protocol):
    """protocol representing a :pep:`249` database connection.

    .. versionadded:: 2.0

    .. seealso::

        `Connection Objects <https://www.python.org/dev/peps/pep-0249/#connection-objects>`_
        - in :pep:`249`

    """

    def close(self) -> None:
        ...

    def commit(self) -> None:
        ...

    def cursor(self) -> DBAPICursor:
        ...

    def rollback(self) -> None:
        ...
    autocommit: bool