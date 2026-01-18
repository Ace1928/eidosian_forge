import flatbuffers
from flatbuffers.compat import import_numpy
class BitcastOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        bitcastOptions = BitcastOptions()
        bitcastOptions.Init(buf, pos)
        return cls.InitFromObj(bitcastOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, bitcastOptions):
        x = BitcastOptionsT()
        x._UnPack(bitcastOptions)
        return x

    def _UnPack(self, bitcastOptions):
        if bitcastOptions is None:
            return

    def Pack(self, builder):
        BitcastOptionsStart(builder)
        bitcastOptions = BitcastOptionsEnd(builder)
        return bitcastOptions