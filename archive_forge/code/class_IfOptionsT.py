import flatbuffers
from flatbuffers.compat import import_numpy
class IfOptionsT(object):

    def __init__(self):
        self.thenSubgraphIndex = 0
        self.elseSubgraphIndex = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        ifOptions = IfOptions()
        ifOptions.Init(buf, pos)
        return cls.InitFromObj(ifOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, ifOptions):
        x = IfOptionsT()
        x._UnPack(ifOptions)
        return x

    def _UnPack(self, ifOptions):
        if ifOptions is None:
            return
        self.thenSubgraphIndex = ifOptions.ThenSubgraphIndex()
        self.elseSubgraphIndex = ifOptions.ElseSubgraphIndex()

    def Pack(self, builder):
        IfOptionsStart(builder)
        IfOptionsAddThenSubgraphIndex(builder, self.thenSubgraphIndex)
        IfOptionsAddElseSubgraphIndex(builder, self.elseSubgraphIndex)
        ifOptions = IfOptionsEnd(builder)
        return ifOptions