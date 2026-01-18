import flatbuffers
from flatbuffers.compat import import_numpy
class CosOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        cosOptions = CosOptions()
        cosOptions.Init(buf, pos)
        return cls.InitFromObj(cosOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, cosOptions):
        x = CosOptionsT()
        x._UnPack(cosOptions)
        return x

    def _UnPack(self, cosOptions):
        if cosOptions is None:
            return

    def Pack(self, builder):
        CosOptionsStart(builder)
        cosOptions = CosOptionsEnd(builder)
        return cosOptions