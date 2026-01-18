from __future__ import annotations
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import no_type_check
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc
from ._typing import insp_is_mapper
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import roles
from ..sql.elements import SQLColumnExpression
from ..sql.elements import SQLCoreOperations
from ..util import FastIntFlag
from ..util.langhelpers import TypingOnly
from ..util.typing import Literal
@no_type_check
def _entity_descriptor(entity: _EntityType[Any], key: str) -> Any:
    """Return a class attribute given an entity and string name.

    May return :class:`.InstrumentedAttribute` or user-defined
    attribute.

    """
    insp = inspection.inspect(entity)
    if insp.is_selectable:
        description = entity
        entity = insp.c
    elif insp.is_aliased_class:
        entity = insp.entity
        description = entity
    elif hasattr(insp, 'mapper'):
        description = entity = insp.mapper.class_
    else:
        description = entity
    try:
        return getattr(entity, key)
    except AttributeError as err:
        raise sa_exc.InvalidRequestError("Entity '%s' has no property '%s'" % (description, key)) from err