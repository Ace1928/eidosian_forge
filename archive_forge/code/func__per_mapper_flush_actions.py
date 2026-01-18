from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import util as orm_util
from .. import event
from .. import util
from ..util import topological
def _per_mapper_flush_actions(self, mapper):
    saves = SaveUpdateAll(self, mapper.base_mapper)
    deletes = DeleteAll(self, mapper.base_mapper)
    self.dependencies.add((saves, deletes))
    for dep in mapper._dependency_processors:
        dep.per_property_preprocessors(self)
    for prop in mapper.relationships:
        if prop.viewonly:
            continue
        dep = prop._dependency_processor
        dep.per_property_preprocessors(self)