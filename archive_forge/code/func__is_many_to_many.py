from ``Engineer`` to ``Employee``, we need to set up both the relationship
from __future__ import annotations
import dataclasses
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..orm import backref
from ..orm import declarative_base as _declarative_base
from ..orm import exc as orm_exc
from ..orm import interfaces
from ..orm import relationship
from ..orm.decl_base import _DeferredMapperConfig
from ..orm.mapper import _CONFIGURE_MUTEX
from ..schema import ForeignKeyConstraint
from ..sql import and_
from ..util import Properties
from ..util.typing import Protocol
def _is_many_to_many(automap_base: Type[Any], table: Table) -> Tuple[Optional[Table], Optional[Table], Optional[list[ForeignKeyConstraint]]]:
    fk_constraints = [const for const in table.constraints if isinstance(const, ForeignKeyConstraint)]
    if len(fk_constraints) != 2:
        return (None, None, None)
    cols: List[Column[Any]] = sum([[fk.parent for fk in fk_constraint.elements] for fk_constraint in fk_constraints], [])
    if set(cols) != set(table.c):
        return (None, None, None)
    return (fk_constraints[0].elements[0].column.table, fk_constraints[1].elements[0].column.table, fk_constraints)