import flatbuffers
from flatbuffers.compat import import_numpy
class ScatterNdOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        scatterNdOptions = ScatterNdOptions()
        scatterNdOptions.Init(buf, pos)
        return cls.InitFromObj(scatterNdOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, scatterNdOptions):
        x = ScatterNdOptionsT()
        x._UnPack(scatterNdOptions)
        return x

    def _UnPack(self, scatterNdOptions):
        if scatterNdOptions is None:
            return

    def Pack(self, builder):
        ScatterNdOptionsStart(builder)
        scatterNdOptions = ScatterNdOptionsEnd(builder)
        return scatterNdOptions