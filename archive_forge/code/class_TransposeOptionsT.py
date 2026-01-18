import flatbuffers
from flatbuffers.compat import import_numpy
class TransposeOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        transposeOptions = TransposeOptions()
        transposeOptions.Init(buf, pos)
        return cls.InitFromObj(transposeOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, transposeOptions):
        x = TransposeOptionsT()
        x._UnPack(transposeOptions)
        return x

    def _UnPack(self, transposeOptions):
        if transposeOptions is None:
            return

    def Pack(self, builder):
        TransposeOptionsStart(builder)
        transposeOptions = TransposeOptionsEnd(builder)
        return transposeOptions