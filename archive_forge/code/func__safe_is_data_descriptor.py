import types
from jedi import debug
def _safe_is_data_descriptor(obj):
    return _safe_hasattr(obj, '__set__') or _safe_hasattr(obj, '__delete__')