import flatbuffers
from flatbuffers.compat import import_numpy
class Conv2DOptionsT(object):

    def __init__(self):
        self.padding = 0
        self.strideW = 0
        self.strideH = 0
        self.fusedActivationFunction = 0
        self.dilationWFactor = 1
        self.dilationHFactor = 1

    @classmethod
    def InitFromBuf(cls, buf, pos):
        conv2Doptions = Conv2DOptions()
        conv2Doptions.Init(buf, pos)
        return cls.InitFromObj(conv2Doptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, conv2Doptions):
        x = Conv2DOptionsT()
        x._UnPack(conv2Doptions)
        return x

    def _UnPack(self, conv2Doptions):
        if conv2Doptions is None:
            return
        self.padding = conv2Doptions.Padding()
        self.strideW = conv2Doptions.StrideW()
        self.strideH = conv2Doptions.StrideH()
        self.fusedActivationFunction = conv2Doptions.FusedActivationFunction()
        self.dilationWFactor = conv2Doptions.DilationWFactor()
        self.dilationHFactor = conv2Doptions.DilationHFactor()

    def Pack(self, builder):
        Conv2DOptionsStart(builder)
        Conv2DOptionsAddPadding(builder, self.padding)
        Conv2DOptionsAddStrideW(builder, self.strideW)
        Conv2DOptionsAddStrideH(builder, self.strideH)
        Conv2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        Conv2DOptionsAddDilationWFactor(builder, self.dilationWFactor)
        Conv2DOptionsAddDilationHFactor(builder, self.dilationHFactor)
        conv2Doptions = Conv2DOptionsEnd(builder)
        return conv2Doptions