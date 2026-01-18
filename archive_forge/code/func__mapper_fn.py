from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import mapperlib as mapperlib
from ._typing import _O
from .descriptor_props import Composite
from .descriptor_props import Synonym
from .interfaces import _AttributeOptions
from .properties import MappedColumn
from .properties import MappedSQLExpression
from .query import AliasOption
from .relationships import _RelationshipArgumentType
from .relationships import _RelationshipDeclared
from .relationships import _RelationshipSecondaryArgument
from .relationships import RelationshipProperty
from .session import Session
from .util import _ORMJoin
from .util import AliasedClass
from .util import AliasedInsp
from .util import LoaderCriteriaOption
from .. import sql
from .. import util
from ..exc import InvalidRequestError
from ..sql._typing import _no_kw
from ..sql.base import _NoArg
from ..sql.base import SchemaEventTarget
from ..sql.schema import _InsertSentinelColumnDefault
from ..sql.schema import SchemaConst
from ..sql.selectable import FromClause
from ..util.typing import Annotated
from ..util.typing import Literal
def _mapper_fn(*arg: Any, **kw: Any) -> NoReturn:
    """Placeholder for the now-removed ``mapper()`` function.

    Classical mappings should be performed using the
    :meth:`_orm.registry.map_imperatively` method.

    This symbol remains in SQLAlchemy 2.0 to suit the deprecated use case
    of using the ``mapper()`` function as a target for ORM event listeners,
    which failed to be marked as deprecated in the 1.4 series.

    Global ORM mapper listeners should instead use the :class:`_orm.Mapper`
    class as the target.

    .. versionchanged:: 2.0  The ``mapper()`` function was removed; the
       symbol remains temporarily as a placeholder for the event listening
       use case.

    """
    raise InvalidRequestError("The 'sqlalchemy.orm.mapper()' function is removed as of SQLAlchemy 2.0.  Use the 'sqlalchemy.orm.registry.map_imperatively()` method of the ``sqlalchemy.orm.registry`` class to perform classical mapping.")