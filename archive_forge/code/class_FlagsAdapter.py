from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class FlagsAdapter(Adapter):
    """
    Adapter for flag fields. Each flag is extracted from the number, resulting
    in a FlagsContainer object. Not intended for direct usage.
    See FlagsEnum.

    Parameters
    * subcon - the subcon to extract
    * flags - a dictionary mapping flag-names to their value
    """
    __slots__ = ['flags']

    def __init__(self, subcon, flags):
        Adapter.__init__(self, subcon)
        self.flags = flags

    def _encode(self, obj, context):
        flags = 0
        for name, value in self.flags.items():
            if getattr(obj, name, False):
                flags |= value
        return flags

    def _decode(self, obj, context):
        obj2 = FlagsContainer()
        for name, value in self.flags.items():
            setattr(obj2, name, bool(obj & value))
        return obj2