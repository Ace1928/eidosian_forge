from __future__ import annotations
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import exc as orm_exc
from . import relationships
from . import util as orm_util
from .base import PassiveFlag
from .query import Query
from .session import object_session
from .writeonly import AbstractCollectionWriter
from .writeonly import WriteOnlyAttributeImpl
from .writeonly import WriteOnlyHistory
from .writeonly import WriteOnlyLoader
from .. import util
from ..engine import result
class AppenderQuery(AppenderMixin[_T], Query[_T]):
    """A dynamic query that supports basic collection storage operations.

    Methods on :class:`.AppenderQuery` include all methods of
    :class:`_orm.Query`, plus additional methods used for collection
    persistence.


    """