import collections
from typing import Any, Callable, List, Tuple, Union
import itertools
def _eq_check(v1: Any, v2: Any) -> bool:
    eq = v1 == v2
    ne = v1 != v2
    assert eq != ne, f'__eq__ is inconsistent with __ne__ between {v1!r} and {v2!r}'
    return eq