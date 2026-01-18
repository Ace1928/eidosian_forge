import sys
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER
import locale
from _pydev_bundle import pydev_log
def _repr_long_iter(self, obj):
    try:
        length = hex(len(obj)) if self.convert_to_hex else len(obj)
        obj_repr = '<%s, len() = %s>' % (type(obj).__name__, length)
    except Exception:
        try:
            obj_repr = '<' + type(obj).__name__ + '>'
        except Exception:
            obj_repr = '<no repr available for object>'
    yield obj_repr