import flatbuffers
from flatbuffers.compat import import_numpy
class GatherNdOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        gatherNdOptions = GatherNdOptions()
        gatherNdOptions.Init(buf, pos)
        return cls.InitFromObj(gatherNdOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, gatherNdOptions):
        x = GatherNdOptionsT()
        x._UnPack(gatherNdOptions)
        return x

    def _UnPack(self, gatherNdOptions):
        if gatherNdOptions is None:
            return

    def Pack(self, builder):
        GatherNdOptionsStart(builder)
        gatherNdOptions = GatherNdOptionsEnd(builder)
        return gatherNdOptions