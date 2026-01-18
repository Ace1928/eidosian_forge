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
def _get_orm_crud_kv_pairs(cls, mapper, statement, kv_iterator, needs_to_be_cacheable):
    core_get_crud_kv_pairs = UpdateDMLState._get_crud_kv_pairs
    for k, v in kv_iterator:
        k = coercions.expect(roles.DMLColumnRole, k)
        if isinstance(k, str):
            desc = _entity_namespace_key(mapper, k, default=NO_VALUE)
            if desc is NO_VALUE:
                yield (coercions.expect(roles.DMLColumnRole, k), coercions.expect(roles.ExpressionElementRole, v, type_=sqltypes.NullType(), is_crud=True) if needs_to_be_cacheable else v)
            else:
                yield from core_get_crud_kv_pairs(statement, desc._bulk_update_tuples(v), needs_to_be_cacheable)
        elif 'entity_namespace' in k._annotations:
            k_anno = k._annotations
            attr = _entity_namespace_key(k_anno['entity_namespace'], k_anno['proxy_key'])
            yield from core_get_crud_kv_pairs(statement, attr._bulk_update_tuples(v), needs_to_be_cacheable)
        else:
            yield (k, v if not needs_to_be_cacheable else coercions.expect(roles.ExpressionElementRole, v, type_=sqltypes.NullType(), is_crud=True))