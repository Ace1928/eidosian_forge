from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class BitIntegerAdapter(Adapter):
    """
    Adapter for bit-integers (converts bitstrings to integers, and vice versa).
    See BitField.

    Parameters:
    * subcon - the subcon to adapt
    * width - the size of the subcon, in bits
    * swapped - whether to swap byte order (little endian/big endian).
      default is False (big endian)
    * signed - whether the value is signed (two's complement). the default
      is False (unsigned)
    * bytesize - number of bits per byte, used for byte-swapping (if swapped).
      default is 8.
    """
    __slots__ = ['width', 'swapped', 'signed', 'bytesize']

    def __init__(self, subcon, width, swapped=False, signed=False, bytesize=8):
        Adapter.__init__(self, subcon)
        self.width = width
        self.swapped = swapped
        self.signed = signed
        self.bytesize = bytesize

    def _encode(self, obj, context):
        if obj < 0 and (not self.signed):
            raise BitIntegerError('object is negative, but field is not signed', obj)
        obj2 = int_to_bin(obj, width=self.width)
        if self.swapped:
            obj2 = swap_bytes(obj2, bytesize=self.bytesize)
        return obj2

    def _decode(self, obj, context):
        if self.swapped:
            obj = swap_bytes(obj, bytesize=self.bytesize)
        return bin_to_int(obj, signed=self.signed)