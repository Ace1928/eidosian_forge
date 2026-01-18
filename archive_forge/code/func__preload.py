import itertools
from functools import update_wrapper
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from .entry_points import load_entry_point
def _preload(self) -> None:
    if self._entry_point is not None:
        load_entry_point(self._entry_point)