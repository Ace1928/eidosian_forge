from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal
@classmethod
def _resolved_keys_as_propnames(cls, mapper, resolved_values):
    values = []
    for k, v in resolved_values:
        if mapper and isinstance(k, expression.ColumnElement):
            try:
                attr = mapper._columntoproperty[k]
            except orm_exc.UnmappedColumnError:
                pass
            else:
                values.append((attr.key, v))
        else:
            raise sa_exc.InvalidRequestError("Attribute name not found, can't be synchronized back to objects: %r" % k)
    return values