import struct
from llvmlite.ir._utils import _StrCaching
def _wrap_packed(self, textrepr):
    """
        Internal helper to wrap textual repr of struct type into packed struct
        """
    if self.packed:
        return '<{}>'.format(textrepr)
    else:
        return textrepr