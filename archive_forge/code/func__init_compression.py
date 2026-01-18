import struct
import builtins
import warnings
from collections import namedtuple
def _init_compression(self):
    if self._comptype == b'G722':
        self._convert = self._lin2adpcm
    elif self._comptype in (b'ulaw', b'ULAW'):
        self._convert = self._lin2ulaw
    elif self._comptype in (b'alaw', b'ALAW'):
        self._convert = self._lin2alaw
    elif self._comptype in (b'sowt', b'SOWT'):
        self._convert = self._lin2sowt