import flatbuffers
from flatbuffers.compat import import_numpy
class ReadVariableOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        readVariableOptions = ReadVariableOptions()
        readVariableOptions.Init(buf, pos)
        return cls.InitFromObj(readVariableOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, readVariableOptions):
        x = ReadVariableOptionsT()
        x._UnPack(readVariableOptions)
        return x

    def _UnPack(self, readVariableOptions):
        if readVariableOptions is None:
            return

    def Pack(self, builder):
        ReadVariableOptionsStart(builder)
        readVariableOptions = ReadVariableOptionsEnd(builder)
        return readVariableOptions