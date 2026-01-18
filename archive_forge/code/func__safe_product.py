import itertools
from typing import Any, Dict, Iterable, List, Tuple
def _safe_product(arrays: List[Iterable[Any]], safe: bool=True) -> Iterable[Tuple]:
    if not safe:
        yield from itertools.product(*arrays)
    else:
        arr = [safe_iter(t) for t in arrays]
        yield from itertools.product(*arr)