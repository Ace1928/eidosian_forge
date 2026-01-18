import flatbuffers
from flatbuffers.compat import import_numpy
class DensifyOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        densifyOptions = DensifyOptions()
        densifyOptions.Init(buf, pos)
        return cls.InitFromObj(densifyOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, densifyOptions):
        x = DensifyOptionsT()
        x._UnPack(densifyOptions)
        return x

    def _UnPack(self, densifyOptions):
        if densifyOptions is None:
            return

    def Pack(self, builder):
        DensifyOptionsStart(builder)
        densifyOptions = DensifyOptionsEnd(builder)
        return densifyOptions