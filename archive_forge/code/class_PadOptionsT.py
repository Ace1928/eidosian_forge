import flatbuffers
from flatbuffers.compat import import_numpy
class PadOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        padOptions = PadOptions()
        padOptions.Init(buf, pos)
        return cls.InitFromObj(padOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, padOptions):
        x = PadOptionsT()
        x._UnPack(padOptions)
        return x

    def _UnPack(self, padOptions):
        if padOptions is None:
            return

    def Pack(self, builder):
        PadOptionsStart(builder)
        padOptions = PadOptionsEnd(builder)
        return padOptions