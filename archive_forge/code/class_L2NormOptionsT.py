import flatbuffers
from flatbuffers.compat import import_numpy
class L2NormOptionsT(object):

    def __init__(self):
        self.fusedActivationFunction = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        l2NormOptions = L2NormOptions()
        l2NormOptions.Init(buf, pos)
        return cls.InitFromObj(l2NormOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, l2NormOptions):
        x = L2NormOptionsT()
        x._UnPack(l2NormOptions)
        return x

    def _UnPack(self, l2NormOptions):
        if l2NormOptions is None:
            return
        self.fusedActivationFunction = l2NormOptions.FusedActivationFunction()

    def Pack(self, builder):
        L2NormOptionsStart(builder)
        L2NormOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        l2NormOptions = L2NormOptionsEnd(builder)
        return l2NormOptions