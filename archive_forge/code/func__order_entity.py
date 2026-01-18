from __future__ import annotations
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import TypeVar
from ..orm.collections import collection
from ..orm.collections import collection_adapter
def _order_entity(self, index, entity, reorder=True):
    have = self._get_order_value(entity)
    if have is not None and (not reorder):
        return
    should_be = self.ordering_func(index, self)
    if have != should_be:
        self._set_order_value(entity, should_be)