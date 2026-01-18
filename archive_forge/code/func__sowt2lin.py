import struct
import builtins
import warnings
from collections import namedtuple
def _sowt2lin(self, data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        import audioop
    return audioop.byteswap(data, 2)