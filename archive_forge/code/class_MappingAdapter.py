from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class MappingAdapter(Adapter):
    """
    Adapter that maps objects to other objects.
    See SymmetricMapping and Enum.

    Parameters:
    * subcon - the subcon to map
    * decoding - the decoding (parsing) mapping (a dict)
    * encoding - the encoding (building) mapping (a dict)
    * decdefault - the default return value when the object is not found
      in the decoding mapping. if no object is given, an exception is raised.
      if `Pass` is used, the unmapped object will be passed as-is
    * encdefault - the default return value when the object is not found
      in the encoding mapping. if no object is given, an exception is raised.
      if `Pass` is used, the unmapped object will be passed as-is
    """
    __slots__ = ['encoding', 'decoding', 'encdefault', 'decdefault']

    def __init__(self, subcon, decoding, encoding, decdefault=NotImplemented, encdefault=NotImplemented):
        Adapter.__init__(self, subcon)
        self.decoding = decoding
        self.encoding = encoding
        self.decdefault = decdefault
        self.encdefault = encdefault

    def _encode(self, obj, context):
        try:
            return self.encoding[obj]
        except (KeyError, TypeError):
            if self.encdefault is NotImplemented:
                raise MappingError('no encoding mapping for %r [%s]' % (obj, self.subcon.name))
            if self.encdefault is Pass:
                return obj
            return self.encdefault

    def _decode(self, obj, context):
        try:
            return self.decoding[obj]
        except (KeyError, TypeError):
            if self.decdefault is NotImplemented:
                raise MappingError('no decoding mapping for %r [%s]' % (obj, self.subcon.name))
            if self.decdefault is Pass:
                return obj
            return self.decdefault