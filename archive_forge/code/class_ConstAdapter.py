from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class ConstAdapter(Adapter):
    """
    Adapter for enforcing a constant value ("magic numbers"). When decoding,
    the return value is checked; when building, the value is substituted in.

    Parameters:
    * subcon - the subcon to validate
    * value - the expected value

    Example:
    Const(Field("signature", 2), "MZ")
    """
    __slots__ = ['value']

    def __init__(self, subcon, value):
        Adapter.__init__(self, subcon)
        self.value = value

    def _encode(self, obj, context):
        if obj is None or obj == self.value:
            return self.value
        else:
            raise ConstError('expected %r, found %r' % (self.value, obj))

    def _decode(self, obj, context):
        if obj != self.value:
            raise ConstError('expected %r, found %r' % (self.value, obj))
        return obj