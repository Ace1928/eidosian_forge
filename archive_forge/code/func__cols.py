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
def _cols(self, mapper: Mapper[_KT]) -> Sequence[ColumnElement[_KT]]:
    cols: List[ColumnElement[_KT]] = []
    metadata = getattr(mapper.local_table, 'metadata', None)
    for ckey, tkey in self.colkeys:
        if tkey is None or metadata is None or tkey not in metadata:
            cols.append(mapper.local_table.c[ckey])
        else:
            cols.append(metadata.tables[tkey].c[ckey])
    return cols