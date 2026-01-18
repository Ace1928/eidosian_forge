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
def _configure_class_instrumentation(self):
    """If this mapper is to be a primary mapper (i.e. the
        non_primary flag is not set), associate this Mapper with the
        given class and entity name.

        Subsequent calls to ``class_mapper()`` for the ``class_`` / ``entity``
        name combination will return this mapper.  Also decorate the
        `__init__` method on the mapped class to include optional
        auto-session attachment logic.

        """
    manager = attributes.opt_manager_of_class(self.class_)
    if self.non_primary:
        if not manager or not manager.is_mapped:
            raise sa_exc.InvalidRequestError('Class %s has no primary mapper configured.  Configure a primary mapper first before setting up a non primary Mapper.' % self.class_)
        self.class_manager = manager
        assert manager.registry is not None
        self.registry = manager.registry
        self._identity_class = manager.mapper._identity_class
        manager.registry._add_non_primary_mapper(self)
        return
    if manager is None or not manager.registry:
        raise sa_exc.InvalidRequestError('The _mapper() function and Mapper() constructor may not be invoked directly outside of a declarative registry. Please use the sqlalchemy.orm.registry.map_imperatively() function for a classical mapping.')
    self.dispatch.instrument_class(self, self.class_)
    manager = instrumentation.register_class(self.class_, mapper=self, expired_attribute_loader=util.partial(loading.load_scalar_attributes, self), finalize=True)
    self.class_manager = manager
    assert manager.registry is not None
    self.registry = manager.registry
    if manager.mapper is None:
        return
    event.listen(manager, 'init', _event_on_init, raw=True)
    for key, method in util.iterate_attributes(self.class_):
        if key == '__init__' and hasattr(method, '_sa_original_init'):
            method = method._sa_original_init
            if hasattr(method, '__func__'):
                method = method.__func__
        if callable(method):
            if hasattr(method, '__sa_reconstructor__'):
                self._reconstructor = method
                event.listen(manager, 'load', _event_on_load, raw=True)
            elif hasattr(method, '__sa_validators__'):
                validation_opts = method.__sa_validation_opts__
                for name in method.__sa_validators__:
                    if name in self.validators:
                        raise sa_exc.InvalidRequestError('A validation function for mapped attribute %r on mapper %s already exists.' % (name, self))
                    self.validators = self.validators.union({name: (method, validation_opts)})