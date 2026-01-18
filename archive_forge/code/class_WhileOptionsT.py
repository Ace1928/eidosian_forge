import flatbuffers
from flatbuffers.compat import import_numpy
class WhileOptionsT(object):

    def __init__(self):
        self.condSubgraphIndex = 0
        self.bodySubgraphIndex = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        whileOptions = WhileOptions()
        whileOptions.Init(buf, pos)
        return cls.InitFromObj(whileOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, whileOptions):
        x = WhileOptionsT()
        x._UnPack(whileOptions)
        return x

    def _UnPack(self, whileOptions):
        if whileOptions is None:
            return
        self.condSubgraphIndex = whileOptions.CondSubgraphIndex()
        self.bodySubgraphIndex = whileOptions.BodySubgraphIndex()

    def Pack(self, builder):
        WhileOptionsStart(builder)
        WhileOptionsAddCondSubgraphIndex(builder, self.condSubgraphIndex)
        WhileOptionsAddBodySubgraphIndex(builder, self.bodySubgraphIndex)
        whileOptions = WhileOptionsEnd(builder)
        return whileOptions