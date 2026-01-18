import struct
import builtins
import warnings
from collections import namedtuple
def _lin2adpcm(self, data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        import audioop
    if not hasattr(self, '_adpcmstate'):
        self._adpcmstate = None
    data, self._adpcmstate = audioop.lin2adpcm(data, 2, self._adpcmstate)
    return data