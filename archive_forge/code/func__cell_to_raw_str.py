from typing import Any, Iterable, List
from triad import Schema
def _cell_to_raw_str(self, obj: Any, truncate_width: int) -> str:
    raw = 'NULL' if obj is None else str(obj)
    if len(raw) > truncate_width:
        raw = raw[:max(0, truncate_width - 3)] + '...'
    return raw