from numba.core.typeconv import castgraph, Conversion
from numba.core import types
class TypeManager(object):
    _conversion_codes = {Conversion.safe: ord('s'), Conversion.unsafe: ord('u'), Conversion.promote: ord('p')}

    def __init__(self):
        self._ptr = _typeconv.new_type_manager()
        self._types = set()

    def select_overload(self, sig, overloads, allow_unsafe, exact_match_required):
        sig = [t._code for t in sig]
        overloads = [[t._code for t in s] for s in overloads]
        return _typeconv.select_overload(self._ptr, sig, overloads, allow_unsafe, exact_match_required)

    def check_compatible(self, fromty, toty):
        if not isinstance(toty, types.Type):
            raise ValueError("Specified type '%s' (%s) is not a Numba type" % (toty, type(toty)))
        name = _typeconv.check_compatible(self._ptr, fromty._code, toty._code)
        conv = Conversion[name] if name is not None else None
        assert conv is not Conversion.nil
        return conv

    def set_compatible(self, fromty, toty, by):
        code = self._conversion_codes[by]
        _typeconv.set_compatible(self._ptr, fromty._code, toty._code, code)
        self._types.add(fromty)
        self._types.add(toty)

    def set_promote(self, fromty, toty):
        self.set_compatible(fromty, toty, Conversion.promote)

    def set_unsafe_convert(self, fromty, toty):
        self.set_compatible(fromty, toty, Conversion.unsafe)

    def set_safe_convert(self, fromty, toty):
        self.set_compatible(fromty, toty, Conversion.safe)

    def get_pointer(self):
        return _typeconv.get_pointer(self._ptr)