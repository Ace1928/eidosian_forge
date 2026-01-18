import flatbuffers
from flatbuffers.compat import import_numpy
class CallOptionsT(object):

    def __init__(self):
        self.subgraph = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        callOptions = CallOptions()
        callOptions.Init(buf, pos)
        return cls.InitFromObj(callOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, callOptions):
        x = CallOptionsT()
        x._UnPack(callOptions)
        return x

    def _UnPack(self, callOptions):
        if callOptions is None:
            return
        self.subgraph = callOptions.Subgraph()

    def Pack(self, builder):
        CallOptionsStart(builder)
        CallOptionsAddSubgraph(builder, self.subgraph)
        callOptions = CallOptionsEnd(builder)
        return callOptions