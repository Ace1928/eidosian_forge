import flatbuffers
from flatbuffers.compat import import_numpy
class ConcatEmbeddingsOptionsT(object):

    def __init__(self):
        self.numChannels = 0
        self.numColumnsPerChannel = None
        self.embeddingDimPerChannel = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        concatEmbeddingsOptions = ConcatEmbeddingsOptions()
        concatEmbeddingsOptions.Init(buf, pos)
        return cls.InitFromObj(concatEmbeddingsOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, concatEmbeddingsOptions):
        x = ConcatEmbeddingsOptionsT()
        x._UnPack(concatEmbeddingsOptions)
        return x

    def _UnPack(self, concatEmbeddingsOptions):
        if concatEmbeddingsOptions is None:
            return
        self.numChannels = concatEmbeddingsOptions.NumChannels()
        if not concatEmbeddingsOptions.NumColumnsPerChannelIsNone():
            if np is None:
                self.numColumnsPerChannel = []
                for i in range(concatEmbeddingsOptions.NumColumnsPerChannelLength()):
                    self.numColumnsPerChannel.append(concatEmbeddingsOptions.NumColumnsPerChannel(i))
            else:
                self.numColumnsPerChannel = concatEmbeddingsOptions.NumColumnsPerChannelAsNumpy()
        if not concatEmbeddingsOptions.EmbeddingDimPerChannelIsNone():
            if np is None:
                self.embeddingDimPerChannel = []
                for i in range(concatEmbeddingsOptions.EmbeddingDimPerChannelLength()):
                    self.embeddingDimPerChannel.append(concatEmbeddingsOptions.EmbeddingDimPerChannel(i))
            else:
                self.embeddingDimPerChannel = concatEmbeddingsOptions.EmbeddingDimPerChannelAsNumpy()

    def Pack(self, builder):
        if self.numColumnsPerChannel is not None:
            if np is not None and type(self.numColumnsPerChannel) is np.ndarray:
                numColumnsPerChannel = builder.CreateNumpyVector(self.numColumnsPerChannel)
            else:
                ConcatEmbeddingsOptionsStartNumColumnsPerChannelVector(builder, len(self.numColumnsPerChannel))
                for i in reversed(range(len(self.numColumnsPerChannel))):
                    builder.PrependInt32(self.numColumnsPerChannel[i])
                numColumnsPerChannel = builder.EndVector()
        if self.embeddingDimPerChannel is not None:
            if np is not None and type(self.embeddingDimPerChannel) is np.ndarray:
                embeddingDimPerChannel = builder.CreateNumpyVector(self.embeddingDimPerChannel)
            else:
                ConcatEmbeddingsOptionsStartEmbeddingDimPerChannelVector(builder, len(self.embeddingDimPerChannel))
                for i in reversed(range(len(self.embeddingDimPerChannel))):
                    builder.PrependInt32(self.embeddingDimPerChannel[i])
                embeddingDimPerChannel = builder.EndVector()
        ConcatEmbeddingsOptionsStart(builder)
        ConcatEmbeddingsOptionsAddNumChannels(builder, self.numChannels)
        if self.numColumnsPerChannel is not None:
            ConcatEmbeddingsOptionsAddNumColumnsPerChannel(builder, numColumnsPerChannel)
        if self.embeddingDimPerChannel is not None:
            ConcatEmbeddingsOptionsAddEmbeddingDimPerChannel(builder, embeddingDimPerChannel)
        concatEmbeddingsOptions = ConcatEmbeddingsOptionsEnd(builder)
        return concatEmbeddingsOptions