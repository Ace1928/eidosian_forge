from io import BytesIO
import struct
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, tostr
from collections import OrderedDict
from collections.abc import MutableMapping
def _readTypeList(self):
    absTypeListOffset = self.absTypeListOffset
    numTypesData = self._read(2, absTypeListOffset)
    self.numTypes, = struct.unpack('>H', numTypesData)
    absTypeListOffset2 = absTypeListOffset + 2
    for i in range(self.numTypes + 1):
        resTypeItemOffset = absTypeListOffset2 + ResourceTypeItemSize * i
        resTypeItemData = self._read(ResourceTypeItemSize, resTypeItemOffset)
        item = sstruct.unpack(ResourceTypeItem, resTypeItemData)
        resType = tostr(item['type'], encoding='mac-roman')
        refListOffset = absTypeListOffset + item['refListOffset']
        numRes = item['numRes'] + 1
        resources = self._readReferenceList(resType, refListOffset, numRes)
        self._resources[resType] = resources