from __future__ import annotations
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import TypeVar
from ..orm.collections import collection
from ..orm.collections import collection_adapter
def _raw_append(self, entity):
    """Append without any ordering behavior."""
    super().append(entity)