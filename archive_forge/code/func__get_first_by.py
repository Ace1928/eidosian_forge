from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
def _get_first_by(self, **kwargs):
    assert hasattr(self, 'get_list')
    entities, _ = self.get_list(**kwargs)
    return entities[0] if entities else None