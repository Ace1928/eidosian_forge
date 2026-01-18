import flatbuffers
from flatbuffers.compat import import_numpy
class RankOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        rankOptions = RankOptions()
        rankOptions.Init(buf, pos)
        return cls.InitFromObj(rankOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, rankOptions):
        x = RankOptionsT()
        x._UnPack(rankOptions)
        return x

    def _UnPack(self, rankOptions):
        if rankOptions is None:
            return

    def Pack(self, builder):
        RankOptionsStart(builder)
        rankOptions = RankOptionsEnd(builder)
        return rankOptions