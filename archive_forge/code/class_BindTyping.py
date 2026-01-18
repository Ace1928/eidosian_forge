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
class BindTyping(Enum):
    """Define different methods of passing typing information for
    bound parameters in a statement to the database driver.

    .. versionadded:: 2.0

    """
    NONE = 1
    'No steps are taken to pass typing information to the database driver.\n\n    This is the default behavior for databases such as SQLite, MySQL / MariaDB,\n    SQL Server.\n\n    '
    SETINPUTSIZES = 2
    'Use the pep-249 setinputsizes method.\n\n    This is only implemented for DBAPIs that support this method and for which\n    the SQLAlchemy dialect has the appropriate infrastructure for that\n    dialect set up.   Current dialects include cx_Oracle as well as\n    optional support for SQL Server using pyodbc.\n\n    When using setinputsizes, dialects also have a means of only using the\n    method for certain datatypes using include/exclude lists.\n\n    When SETINPUTSIZES is used, the :meth:`.Dialect.do_set_input_sizes` method\n    is called for each statement executed which has bound parameters.\n\n    '
    RENDER_CASTS = 3
    'Render casts or other directives in the SQL string.\n\n    This method is used for all PostgreSQL dialects, including asyncpg,\n    pg8000, psycopg, psycopg2.   Dialects which implement this can choose\n    which kinds of datatypes are explicitly cast in SQL statements and which\n    aren\'t.\n\n    When RENDER_CASTS is used, the compiler will invoke the\n    :meth:`.SQLCompiler.render_bind_cast` method for the rendered\n    string representation of each :class:`.BindParameter` object whose\n    dialect-level type sets the :attr:`.TypeEngine.render_bind_cast` attribute.\n\n    The :meth:`.SQLCompiler.render_bind_cast` is also used to render casts\n    for one form of "insertmanyvalues" query, when both\n    :attr:`.InsertmanyvaluesSentinelOpts.USE_INSERT_FROM_SELECT` and\n    :attr:`.InsertmanyvaluesSentinelOpts.RENDER_SELECT_COL_CASTS` are set,\n    where the casts are applied to the intermediary columns e.g.\n    "INSERT INTO t (a, b, c) SELECT p0::TYP, p1::TYP, p2::TYP "\n    "FROM (VALUES (?, ?), (?, ?), ...)".\n\n    .. versionadded:: 2.0.10 - :meth:`.SQLCompiler.render_bind_cast` is now\n       used within some elements of the "insertmanyvalues" implementation.\n\n\n    '