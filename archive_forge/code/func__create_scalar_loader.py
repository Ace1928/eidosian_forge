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
def _create_scalar_loader(self, context, key, _instance, populators):

    def load_scalar_from_joined_new_row(state, dict_, row):
        dict_[key] = _instance(row)

    def load_scalar_from_joined_existing_row(state, dict_, row):
        existing = _instance(row)
        if key in dict_:
            if existing is not dict_[key]:
                util.warn("Multiple rows returned with uselist=False for eagerly-loaded attribute '%s' " % self)
        else:
            dict_[key] = existing

    def load_scalar_from_joined_exec(state, dict_, row):
        _instance(row)
    populators['new'].append((self.key, load_scalar_from_joined_new_row))
    populators['existing'].append((self.key, load_scalar_from_joined_existing_row))
    if context.invoke_all_eagers:
        populators['eager'].append((self.key, load_scalar_from_joined_exec))