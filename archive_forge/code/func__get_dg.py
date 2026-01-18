from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
def _get_dg(self, index: Optional[int]) -> DeltaGenerator:
    if index is not None:
        assert 0 <= index < len(self._child_records), f'Bad index: {index}'
        return self._child_records[index].dg
    return self._container