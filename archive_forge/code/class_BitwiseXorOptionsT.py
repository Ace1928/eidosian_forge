import flatbuffers
from flatbuffers.compat import import_numpy
class BitwiseXorOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        bitwiseXorOptions = BitwiseXorOptions()
        bitwiseXorOptions.Init(buf, pos)
        return cls.InitFromObj(bitwiseXorOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, bitwiseXorOptions):
        x = BitwiseXorOptionsT()
        x._UnPack(bitwiseXorOptions)
        return x

    def _UnPack(self, bitwiseXorOptions):
        if bitwiseXorOptions is None:
            return

    def Pack(self, builder):
        BitwiseXorOptionsStart(builder)
        bitwiseXorOptions = BitwiseXorOptionsEnd(builder)
        return bitwiseXorOptions