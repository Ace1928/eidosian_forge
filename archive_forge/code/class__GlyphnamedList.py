from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, strjoin
from . import DefaultTable
import array
from collections.abc import Mapping
class _GlyphnamedList(Mapping):

    def __init__(self, reverseGlyphOrder, data):
        self._array = data
        self._map = dict(reverseGlyphOrder)

    def __getitem__(self, k):
        return self._array[self._map[k]]

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def keys(self):
        return self._map.keys()