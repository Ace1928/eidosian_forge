from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class StringAdapter(Adapter):
    """
    Adapter for strings. Converts a sequence of characters into a python
    string, and optionally handles character encoding.
    See String.

    Parameters:
    * subcon - the subcon to convert
    * encoding - the character encoding name (e.g., "utf8"), or None to
      return raw bytes (usually 8-bit ASCII).
    """
    __slots__ = ['encoding']

    def __init__(self, subcon, encoding=None):
        Adapter.__init__(self, subcon)
        self.encoding = encoding

    def _encode(self, obj, context):
        if self.encoding:
            obj = obj.encode(self.encoding)
        return obj

    def _decode(self, obj, context):
        if self.encoding:
            obj = obj.decode(self.encoding)
        return obj