import flatbuffers
from flatbuffers.compat import import_numpy
class MatrixDiagOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        matrixDiagOptions = MatrixDiagOptions()
        matrixDiagOptions.Init(buf, pos)
        return cls.InitFromObj(matrixDiagOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, matrixDiagOptions):
        x = MatrixDiagOptionsT()
        x._UnPack(matrixDiagOptions)
        return x

    def _UnPack(self, matrixDiagOptions):
        if matrixDiagOptions is None:
            return

    def Pack(self, builder):
        MatrixDiagOptionsStart(builder)
        matrixDiagOptions = MatrixDiagOptionsEnd(builder)
        return matrixDiagOptions