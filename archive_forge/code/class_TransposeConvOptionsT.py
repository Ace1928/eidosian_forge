import flatbuffers
from flatbuffers.compat import import_numpy
class TransposeConvOptionsT(object):

    def __init__(self):
        self.padding = 0
        self.strideW = 0
        self.strideH = 0
        self.fusedActivationFunction = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        transposeConvOptions = TransposeConvOptions()
        transposeConvOptions.Init(buf, pos)
        return cls.InitFromObj(transposeConvOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, transposeConvOptions):
        x = TransposeConvOptionsT()
        x._UnPack(transposeConvOptions)
        return x

    def _UnPack(self, transposeConvOptions):
        if transposeConvOptions is None:
            return
        self.padding = transposeConvOptions.Padding()
        self.strideW = transposeConvOptions.StrideW()
        self.strideH = transposeConvOptions.StrideH()
        self.fusedActivationFunction = transposeConvOptions.FusedActivationFunction()

    def Pack(self, builder):
        TransposeConvOptionsStart(builder)
        TransposeConvOptionsAddPadding(builder, self.padding)
        TransposeConvOptionsAddStrideW(builder, self.strideW)
        TransposeConvOptionsAddStrideH(builder, self.strideH)
        TransposeConvOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        transposeConvOptions = TransposeConvOptionsEnd(builder)
        return transposeConvOptions