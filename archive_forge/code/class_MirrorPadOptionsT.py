import flatbuffers
from flatbuffers.compat import import_numpy
class MirrorPadOptionsT(object):

    def __init__(self):
        self.mode = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        mirrorPadOptions = MirrorPadOptions()
        mirrorPadOptions.Init(buf, pos)
        return cls.InitFromObj(mirrorPadOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, mirrorPadOptions):
        x = MirrorPadOptionsT()
        x._UnPack(mirrorPadOptions)
        return x

    def _UnPack(self, mirrorPadOptions):
        if mirrorPadOptions is None:
            return
        self.mode = mirrorPadOptions.Mode()

    def Pack(self, builder):
        MirrorPadOptionsStart(builder)
        MirrorPadOptionsAddMode(builder, self.mode)
        mirrorPadOptions = MirrorPadOptionsEnd(builder)
        return mirrorPadOptions