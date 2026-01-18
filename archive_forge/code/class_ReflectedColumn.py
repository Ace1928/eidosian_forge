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
class ReflectedColumn(TypedDict):
    """Dictionary representing the reflected elements corresponding to
    a :class:`_schema.Column` object.

    The :class:`.ReflectedColumn` structure is returned by the
    :class:`.Inspector.get_columns` method.

    """
    name: str
    'column name'
    type: TypeEngine[Any]
    'column type represented as a :class:`.TypeEngine` instance.'
    nullable: bool
    'boolean flag if the column is NULL or NOT NULL'
    default: Optional[str]
    'column default expression as a SQL string'
    autoincrement: NotRequired[bool]
    'database-dependent autoincrement flag.\n\n    This flag indicates if the column has a database-side "autoincrement"\n    flag of some kind.   Within SQLAlchemy, other kinds of columns may\n    also act as an "autoincrement" column without necessarily having\n    such a flag on them.\n\n    See :paramref:`_schema.Column.autoincrement` for more background on\n    "autoincrement".\n\n    '
    comment: NotRequired[Optional[str]]
    'comment for the column, if present.\n    Only some dialects return this key\n    '
    computed: NotRequired[ReflectedComputed]
    'indicates that this column is computed by the database.\n    Only some dialects return this key.\n\n    .. versionadded:: 1.3.16 - added support for computed reflection.\n    '
    identity: NotRequired[ReflectedIdentity]
    'indicates this column is an IDENTITY column.\n    Only some dialects return this key.\n\n    .. versionadded:: 1.4 - added support for identity column reflection.\n    '
    dialect_options: NotRequired[Dict[str, Any]]
    'Additional dialect-specific options detected for this reflected\n    object'