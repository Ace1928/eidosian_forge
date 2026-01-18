from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
class WOFFFlavorData:
    Flavor = 'woff'

    def __init__(self, reader=None):
        self.majorVersion = None
        self.minorVersion = None
        self.metaData = None
        self.privData = None
        if reader:
            self.majorVersion = reader.majorVersion
            self.minorVersion = reader.minorVersion
            if reader.metaLength:
                reader.file.seek(reader.metaOffset)
                rawData = reader.file.read(reader.metaLength)
                assert len(rawData) == reader.metaLength
                data = self._decompress(rawData)
                assert len(data) == reader.metaOrigLength
                self.metaData = data
            if reader.privLength:
                reader.file.seek(reader.privOffset)
                data = reader.file.read(reader.privLength)
                assert len(data) == reader.privLength
                self.privData = data

    def _decompress(self, rawData):
        import zlib
        return zlib.decompress(rawData)