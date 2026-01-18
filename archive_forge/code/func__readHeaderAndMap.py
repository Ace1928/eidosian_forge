from io import BytesIO
import struct
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, tostr
from collections import OrderedDict
from collections.abc import MutableMapping
def _readHeaderAndMap(self):
    self.file.seek(0)
    headerData = self._read(ResourceForkHeaderSize)
    sstruct.unpack(ResourceForkHeader, headerData, self)
    mapOffset = self.mapOffset + 22
    resourceMapData = self._read(ResourceMapHeaderSize, mapOffset)
    sstruct.unpack(ResourceMapHeader, resourceMapData, self)
    self.absTypeListOffset = self.mapOffset + self.typeListOffset
    self.absNameListOffset = self.mapOffset + self.nameListOffset