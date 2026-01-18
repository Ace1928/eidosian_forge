import re
from .errors import JSONDecodeError
def _import_c_make_scanner():
    try:
        from ._speedups import make_scanner
        return make_scanner
    except ImportError:
        return None