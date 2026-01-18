import builtins
from itertools import islice
from _thread import get_ident
def _possibly_sorted(x):
    try:
        return sorted(x)
    except Exception:
        return list(x)