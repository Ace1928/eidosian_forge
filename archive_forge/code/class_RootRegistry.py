from __future__ import annotations
from functools import reduce
from itertools import chain
import logging
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import base as orm_base
from ._typing import insp_is_mapper_property
from .. import exc
from .. import util
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
class RootRegistry(CreatesToken):
    """Root registry, defers to mappers so that
    paths are maintained per-root-mapper.

    """
    __slots__ = ()
    inherit_cache = True
    path = natural_path = ()
    has_entity = False
    is_aliased_class = False
    is_root = True
    is_unnatural = False

    def _getitem(self, entity: Any) -> Union[TokenRegistry, AbstractEntityRegistry]:
        if entity in PathToken._intern:
            if TYPE_CHECKING:
                assert isinstance(entity, _StrPathToken)
            return TokenRegistry(self, PathToken._intern[entity])
        else:
            try:
                return entity._path_registry
            except AttributeError:
                raise IndexError(f'invalid argument for RootRegistry.__getitem__: {entity}')

    def _truncate_recursive(self) -> RootRegistry:
        return self
    if not TYPE_CHECKING:
        __getitem__ = _getitem