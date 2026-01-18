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
class LoadLazyAttribute:
    """semi-serializable loader object used by LazyLoader

    Historically, this object would be carried along with instances that
    needed to run lazyloaders, so it had to be serializable to support
    cached instances.

    this is no longer a general requirement, and the case where this object
    is used is exactly the case where we can't really serialize easily,
    which is when extra criteria in the loader option is present.

    We can't reliably serialize that as it refers to mapped entities and
    AliasedClass objects that are local to the current process, which would
    need to be matched up on deserialize e.g. the sqlalchemy.ext.serializer
    approach.

    """

    def __init__(self, key, initiating_strategy, loadopt, extra_criteria):
        self.key = key
        self.strategy_key = initiating_strategy.strategy_key
        self.loadopt = loadopt
        self.extra_criteria = extra_criteria

    def __getstate__(self):
        if self.extra_criteria is not None:
            util.warn("Can't reliably serialize a lazyload() option that contains additional criteria; please use eager loading for this case")
        return {'key': self.key, 'strategy_key': self.strategy_key, 'loadopt': self.loadopt, 'extra_criteria': ()}

    def __call__(self, state, passive=attributes.PASSIVE_OFF):
        key = self.key
        instance_mapper = state.manager.mapper
        prop = instance_mapper._props[key]
        strategy = prop._strategies[self.strategy_key]
        return strategy._load_for_state(state, passive, loadopt=self.loadopt, extra_criteria=self.extra_criteria)