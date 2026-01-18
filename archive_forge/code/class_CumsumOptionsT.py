import flatbuffers
from flatbuffers.compat import import_numpy
class CumsumOptionsT(object):

    def __init__(self):
        self.exclusive = False
        self.reverse = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        cumsumOptions = CumsumOptions()
        cumsumOptions.Init(buf, pos)
        return cls.InitFromObj(cumsumOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, cumsumOptions):
        x = CumsumOptionsT()
        x._UnPack(cumsumOptions)
        return x

    def _UnPack(self, cumsumOptions):
        if cumsumOptions is None:
            return
        self.exclusive = cumsumOptions.Exclusive()
        self.reverse = cumsumOptions.Reverse()

    def Pack(self, builder):
        CumsumOptionsStart(builder)
        CumsumOptionsAddExclusive(builder, self.exclusive)
        CumsumOptionsAddReverse(builder, self.reverse)
        cumsumOptions = CumsumOptionsEnd(builder)
        return cumsumOptions