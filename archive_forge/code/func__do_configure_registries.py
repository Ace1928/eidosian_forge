from __future__ import annotations
from collections import deque
from functools import reduce
from itertools import chain
import sys
import threading
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import exc as orm_exc
from . import instrumentation
from . import loading
from . import properties
from . import util as orm_util
from ._typing import _O
from .base import _class_to_mapper
from .base import _parse_mapper_argument
from .base import _state_mapper
from .base import PassiveFlag
from .base import state_str
from .interfaces import _MappedAttribute
from .interfaces import EXT_SKIP
from .interfaces import InspectionAttr
from .interfaces import MapperProperty
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .interfaces import StrategizedProperty
from .path_registry import PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import TableClause
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import KeyedColumnElement
from ..sql.schema import Column
from ..sql.schema import Table
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import HasMemoized
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
@util.preload_module('sqlalchemy.orm.decl_api')
def _do_configure_registries(registries: Set[_RegistryType], cascade: bool) -> None:
    registry = util.preloaded.orm_decl_api.registry
    orig = set(registries)
    for reg in registry._recurse_with_dependencies(registries):
        has_skip = False
        for mapper in reg._mappers_to_configure():
            run_configure = None
            for fn in mapper.dispatch.before_mapper_configured:
                run_configure = fn(mapper, mapper.class_)
                if run_configure is EXT_SKIP:
                    has_skip = True
                    break
            if run_configure is EXT_SKIP:
                continue
            if getattr(mapper, '_configure_failed', False):
                e = sa_exc.InvalidRequestError("One or more mappers failed to initialize - can't proceed with initialization of other mappers. Triggering mapper: '%s'. Original exception was: %s" % (mapper, mapper._configure_failed))
                e._configure_failed = mapper._configure_failed
                raise e
            if not mapper.configured:
                try:
                    mapper._post_configure_properties()
                    mapper._expire_memoizations()
                    mapper.dispatch.mapper_configured(mapper, mapper.class_)
                except Exception:
                    exc = sys.exc_info()[1]
                    if not hasattr(exc, '_configure_failed'):
                        mapper._configure_failed = exc
                    raise
        if not has_skip:
            reg._new_mappers = False
        if not cascade and reg._dependencies.difference(orig):
            raise sa_exc.InvalidRequestError('configure was called with cascade=False but additional registries remain')