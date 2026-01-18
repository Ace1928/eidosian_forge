import struct
import builtins
import warnings
from collections import namedtuple
def _lin2alaw(self, data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        import audioop
    return audioop.lin2alaw(data, 2)