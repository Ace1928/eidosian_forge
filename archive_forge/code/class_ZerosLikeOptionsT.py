import flatbuffers
from flatbuffers.compat import import_numpy
class ZerosLikeOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        zerosLikeOptions = ZerosLikeOptions()
        zerosLikeOptions.Init(buf, pos)
        return cls.InitFromObj(zerosLikeOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, zerosLikeOptions):
        x = ZerosLikeOptionsT()
        x._UnPack(zerosLikeOptions)
        return x

    def _UnPack(self, zerosLikeOptions):
        if zerosLikeOptions is None:
            return

    def Pack(self, builder):
        ZerosLikeOptionsStart(builder)
        zerosLikeOptions = ZerosLikeOptionsEnd(builder)
        return zerosLikeOptions