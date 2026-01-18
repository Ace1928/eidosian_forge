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
def _generate_natural_for_superclasses(self) -> Iterator[_PathRepresentation]:
    parent = self.parent
    if is_root(parent):
        yield self.natural_path
        return
    if TYPE_CHECKING:
        assert isinstance(parent, AbstractEntityRegistry)
    for mp_ent in parent.mapper.iterate_to_root():
        yield TokenRegistry(parent.parent[mp_ent], self.token).natural_path
    if parent.is_aliased_class and cast('AliasedInsp[Any]', parent.entity)._is_with_polymorphic:
        yield self.natural_path
        for ent in cast('AliasedInsp[Any]', parent.entity)._with_polymorphic_entities:
            yield TokenRegistry(parent.parent[ent], self.token).natural_path
    else:
        yield self.natural_path