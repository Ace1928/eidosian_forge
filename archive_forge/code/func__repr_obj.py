import sys
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER
import locale
from _pydev_bundle import pydev_log
def _repr_obj(self, obj, level, limit_inner, limit_outer):
    try:
        if self.raw_value:
            if isinstance(obj, bytes):
                yield obj.decode('latin-1')
                return
            try:
                mv = memoryview(obj)
            except Exception:
                yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                return
            else:
                yield mv.tobytes().decode('latin-1')
                return
        elif self.convert_to_hex and isinstance(obj, self.int_types):
            obj_repr = hex(obj)
        else:
            obj_repr = repr(obj)
    except Exception:
        try:
            obj_repr = object.__repr__(obj)
        except Exception:
            try:
                obj_repr = '<no repr available for ' + type(obj).__name__ + '>'
            except Exception:
                obj_repr = '<no repr available for object>'
    limit = limit_inner if level > 0 else limit_outer
    if limit >= len(obj_repr):
        yield self._convert_to_unicode_or_bytes_repr(obj_repr)
        return
    left_count, right_count = (max(1, int(2 * limit / 3)), max(1, int(limit / 3)))
    yield obj_repr[:left_count]
    yield '...'
    yield obj_repr[-right_count:]