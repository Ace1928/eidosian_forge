import flatbuffers
from flatbuffers.compat import import_numpy
class MaximumMinimumOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        maximumMinimumOptions = MaximumMinimumOptions()
        maximumMinimumOptions.Init(buf, pos)
        return cls.InitFromObj(maximumMinimumOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, maximumMinimumOptions):
        x = MaximumMinimumOptionsT()
        x._UnPack(maximumMinimumOptions)
        return x

    def _UnPack(self, maximumMinimumOptions):
        if maximumMinimumOptions is None:
            return

    def Pack(self, builder):
        MaximumMinimumOptionsStart(builder)
        maximumMinimumOptions = MaximumMinimumOptionsEnd(builder)
        return maximumMinimumOptions