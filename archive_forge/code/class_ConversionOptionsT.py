import flatbuffers
from flatbuffers.compat import import_numpy
class ConversionOptionsT(object):

    def __init__(self):
        self.modelOptimizationModes = None
        self.allowCustomOps = False
        self.enableSelectTfOps = False
        self.forceSelectTfOps = False
        self.sparsityBlockSizes = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        conversionOptions = ConversionOptions()
        conversionOptions.Init(buf, pos)
        return cls.InitFromObj(conversionOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, conversionOptions):
        x = ConversionOptionsT()
        x._UnPack(conversionOptions)
        return x

    def _UnPack(self, conversionOptions):
        if conversionOptions is None:
            return
        if not conversionOptions.ModelOptimizationModesIsNone():
            if np is None:
                self.modelOptimizationModes = []
                for i in range(conversionOptions.ModelOptimizationModesLength()):
                    self.modelOptimizationModes.append(conversionOptions.ModelOptimizationModes(i))
            else:
                self.modelOptimizationModes = conversionOptions.ModelOptimizationModesAsNumpy()
        self.allowCustomOps = conversionOptions.AllowCustomOps()
        self.enableSelectTfOps = conversionOptions.EnableSelectTfOps()
        self.forceSelectTfOps = conversionOptions.ForceSelectTfOps()
        if not conversionOptions.SparsityBlockSizesIsNone():
            self.sparsityBlockSizes = []
            for i in range(conversionOptions.SparsityBlockSizesLength()):
                if conversionOptions.SparsityBlockSizes(i) is None:
                    self.sparsityBlockSizes.append(None)
                else:
                    sparsityBlockSize_ = SparsityBlockSizeT.InitFromObj(conversionOptions.SparsityBlockSizes(i))
                    self.sparsityBlockSizes.append(sparsityBlockSize_)

    def Pack(self, builder):
        if self.modelOptimizationModes is not None:
            if np is not None and type(self.modelOptimizationModes) is np.ndarray:
                modelOptimizationModes = builder.CreateNumpyVector(self.modelOptimizationModes)
            else:
                ConversionOptionsStartModelOptimizationModesVector(builder, len(self.modelOptimizationModes))
                for i in reversed(range(len(self.modelOptimizationModes))):
                    builder.PrependInt32(self.modelOptimizationModes[i])
                modelOptimizationModes = builder.EndVector()
        if self.sparsityBlockSizes is not None:
            sparsityBlockSizeslist = []
            for i in range(len(self.sparsityBlockSizes)):
                sparsityBlockSizeslist.append(self.sparsityBlockSizes[i].Pack(builder))
            ConversionOptionsStartSparsityBlockSizesVector(builder, len(self.sparsityBlockSizes))
            for i in reversed(range(len(self.sparsityBlockSizes))):
                builder.PrependUOffsetTRelative(sparsityBlockSizeslist[i])
            sparsityBlockSizes = builder.EndVector()
        ConversionOptionsStart(builder)
        if self.modelOptimizationModes is not None:
            ConversionOptionsAddModelOptimizationModes(builder, modelOptimizationModes)
        ConversionOptionsAddAllowCustomOps(builder, self.allowCustomOps)
        ConversionOptionsAddEnableSelectTfOps(builder, self.enableSelectTfOps)
        ConversionOptionsAddForceSelectTfOps(builder, self.forceSelectTfOps)
        if self.sparsityBlockSizes is not None:
            ConversionOptionsAddSparsityBlockSizes(builder, sparsityBlockSizes)
        conversionOptions = ConversionOptionsEnd(builder)
        return conversionOptions