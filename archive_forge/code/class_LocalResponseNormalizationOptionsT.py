import flatbuffers
from flatbuffers.compat import import_numpy
class LocalResponseNormalizationOptionsT(object):

    def __init__(self):
        self.radius = 0
        self.bias = 0.0
        self.alpha = 0.0
        self.beta = 0.0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        localResponseNormalizationOptions = LocalResponseNormalizationOptions()
        localResponseNormalizationOptions.Init(buf, pos)
        return cls.InitFromObj(localResponseNormalizationOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, localResponseNormalizationOptions):
        x = LocalResponseNormalizationOptionsT()
        x._UnPack(localResponseNormalizationOptions)
        return x

    def _UnPack(self, localResponseNormalizationOptions):
        if localResponseNormalizationOptions is None:
            return
        self.radius = localResponseNormalizationOptions.Radius()
        self.bias = localResponseNormalizationOptions.Bias()
        self.alpha = localResponseNormalizationOptions.Alpha()
        self.beta = localResponseNormalizationOptions.Beta()

    def Pack(self, builder):
        LocalResponseNormalizationOptionsStart(builder)
        LocalResponseNormalizationOptionsAddRadius(builder, self.radius)
        LocalResponseNormalizationOptionsAddBias(builder, self.bias)
        LocalResponseNormalizationOptionsAddAlpha(builder, self.alpha)
        LocalResponseNormalizationOptionsAddBeta(builder, self.beta)
        localResponseNormalizationOptions = LocalResponseNormalizationOptionsEnd(builder)
        return localResponseNormalizationOptions