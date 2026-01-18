import flatbuffers
from flatbuffers.compat import import_numpy
class LessEqualOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        lessEqualOptions = LessEqualOptions()
        lessEqualOptions.Init(buf, pos)
        return cls.InitFromObj(lessEqualOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, lessEqualOptions):
        x = LessEqualOptionsT()
        x._UnPack(lessEqualOptions)
        return x

    def _UnPack(self, lessEqualOptions):
        if lessEqualOptions is None:
            return

    def Pack(self, builder):
        LessEqualOptionsStart(builder)
        lessEqualOptions = LessEqualOptionsEnd(builder)
        return lessEqualOptions