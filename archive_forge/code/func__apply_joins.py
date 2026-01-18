from __future__ import annotations
import collections
import itertools
from typing import Any
from typing import Dict
from typing import Tuple
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import interfaces
from . import loading
from . import path_registry
from . import properties
from . import query
from . import relationships
from . import unitofwork
from . import util as orm_util
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import ATTR_WAS_SET
from .base import LoaderCallableStatus
from .base import PASSIVE_OFF
from .base import PassiveFlag
from .context import _column_descriptions
from .context import ORMCompileState
from .context import ORMSelectCompileState
from .context import QueryContext
from .interfaces import LoaderStrategy
from .interfaces import StrategizedProperty
from .session import _state_session
from .state import InstanceState
from .strategy_options import Load
from .util import _none_set
from .util import AliasedClass
from .. import event
from .. import exc as sa_exc
from .. import inspect
from .. import log
from .. import sql
from .. import util
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
def _apply_joins(self, q, to_join, left_alias, parent_alias, effective_entity):
    ltj = len(to_join)
    if ltj == 1:
        to_join = [getattr(left_alias, to_join[0][1]).of_type(effective_entity)]
    elif ltj == 2:
        to_join = [getattr(left_alias, to_join[0][1]).of_type(parent_alias), getattr(parent_alias, to_join[-1][1]).of_type(effective_entity)]
    elif ltj > 2:
        middle = [(orm_util.AliasedClass(item[0]) if not inspect(item[0]).is_aliased_class else item[0].entity, item[1]) for item in to_join[1:-1]]
        inner = []
        while middle:
            item = middle.pop(0)
            attr = getattr(item[0], item[1])
            if middle:
                attr = attr.of_type(middle[0][0])
            else:
                attr = attr.of_type(parent_alias)
            inner.append(attr)
        to_join = [getattr(left_alias, to_join[0][1]).of_type(inner[0].parent)] + inner + [getattr(parent_alias, to_join[-1][1]).of_type(effective_entity)]
    for attr in to_join:
        q = q.join(attr)
    return q