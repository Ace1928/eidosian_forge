from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class PaddingAdapter(Adapter):
    """
    Adapter for padding.

    Parameters:
    * subcon - the subcon to pad
    * pattern - the padding pattern (character as byte). default is b"\\x00"
    * strict - whether or not to verify, during parsing, that the given
      padding matches the padding pattern. default is False (unstrict)
    """
    __slots__ = ['pattern', 'strict']

    def __init__(self, subcon, pattern=b'\x00', strict=False):
        Adapter.__init__(self, subcon)
        self.pattern = pattern
        self.strict = strict

    def _encode(self, obj, context):
        return self._sizeof(context) * self.pattern

    def _decode(self, obj, context):
        if self.strict:
            expected = self._sizeof(context) * self.pattern
            if obj != expected:
                raise PaddingError('expected %r, found %r' % (expected, obj))
        return obj