from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import base
from .collections import collection
from .collections import collection_adapter
from .. import exc as sa_exc
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..util.typing import Literal
@classmethod
def _reduce_from_cols(cls, cols: Sequence[ColumnElement[_KT]]) -> Tuple[Type[_SerializableColumnGetterV2[_KT]], Tuple[Sequence[Tuple[Optional[str], Optional[str]]]]]:

    def _table_key(c: ColumnElement[_KT]) -> Optional[str]:
        if not isinstance(c.table, expression.TableClause):
            return None
        else:
            return c.table.key
    colkeys = [(c.key, _table_key(c)) for c in cols]
    return (_SerializableColumnGetterV2, (colkeys,))