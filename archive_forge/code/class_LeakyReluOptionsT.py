import flatbuffers
from flatbuffers.compat import import_numpy
class LeakyReluOptionsT(object):

    def __init__(self):
        self.alpha = 0.0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        leakyReluOptions = LeakyReluOptions()
        leakyReluOptions.Init(buf, pos)
        return cls.InitFromObj(leakyReluOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, leakyReluOptions):
        x = LeakyReluOptionsT()
        x._UnPack(leakyReluOptions)
        return x

    def _UnPack(self, leakyReluOptions):
        if leakyReluOptions is None:
            return
        self.alpha = leakyReluOptions.Alpha()

    def Pack(self, builder):
        LeakyReluOptionsStart(builder)
        LeakyReluOptionsAddAlpha(builder, self.alpha)
        leakyReluOptions = LeakyReluOptionsEnd(builder)
        return leakyReluOptions