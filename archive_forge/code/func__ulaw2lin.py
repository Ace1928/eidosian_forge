import struct
import builtins
import warnings
from collections import namedtuple
def _ulaw2lin(self, data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        import audioop
    return audioop.ulaw2lin(data, 2)