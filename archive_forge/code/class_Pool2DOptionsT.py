import flatbuffers
from flatbuffers.compat import import_numpy
class Pool2DOptionsT(object):

    def __init__(self):
        self.padding = 0
        self.strideW = 0
        self.strideH = 0
        self.filterWidth = 0
        self.filterHeight = 0
        self.fusedActivationFunction = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        pool2Doptions = Pool2DOptions()
        pool2Doptions.Init(buf, pos)
        return cls.InitFromObj(pool2Doptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, pool2Doptions):
        x = Pool2DOptionsT()
        x._UnPack(pool2Doptions)
        return x

    def _UnPack(self, pool2Doptions):
        if pool2Doptions is None:
            return
        self.padding = pool2Doptions.Padding()
        self.strideW = pool2Doptions.StrideW()
        self.strideH = pool2Doptions.StrideH()
        self.filterWidth = pool2Doptions.FilterWidth()
        self.filterHeight = pool2Doptions.FilterHeight()
        self.fusedActivationFunction = pool2Doptions.FusedActivationFunction()

    def Pack(self, builder):
        Pool2DOptionsStart(builder)
        Pool2DOptionsAddPadding(builder, self.padding)
        Pool2DOptionsAddStrideW(builder, self.strideW)
        Pool2DOptionsAddStrideH(builder, self.strideH)
        Pool2DOptionsAddFilterWidth(builder, self.filterWidth)
        Pool2DOptionsAddFilterHeight(builder, self.filterHeight)
        Pool2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        pool2Doptions = Pool2DOptionsEnd(builder)
        return pool2Doptions