from numba.core.typeconv import castgraph, Conversion
from numba.core import types
def _cb_update(self, a, b, rel):
    """
        Callback for updating.
        """
    if rel == Conversion.promote:
        self._tm.set_promote(a, b)
    elif rel == Conversion.safe:
        self._tm.set_safe_convert(a, b)
    elif rel == Conversion.unsafe:
        self._tm.set_unsafe_convert(a, b)
    else:
        raise AssertionError(rel)