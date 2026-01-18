from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class LengthValueAdapter(Adapter):
    """
    Adapter for length-value pairs. It extracts only the value from the
    pair, and calculates the length based on the value.
    See PrefixedArray and PascalString.

    Parameters:
    * subcon - the subcon returning a length-value pair
    """
    __slots__ = []

    def _encode(self, obj, context):
        return (len(obj), obj)

    def _decode(self, obj, context):
        return obj[1]