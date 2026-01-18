from typing import (
import cmath
import re
import numpy as np
import sympy
def _segment_by(seq: Iterable[T], *, key: Callable[[T], Any]) -> Iterator[List[T]]:
    group: List[T] = []
    last_key = None
    for item in seq:
        item_key = key(item)
        if len(group) and item_key != last_key:
            yield group
            group = []
        group.append(item)
        last_key = item_key
    if len(group) > 0:
        yield group