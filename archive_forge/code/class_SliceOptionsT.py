import flatbuffers
from flatbuffers.compat import import_numpy
class SliceOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sliceOptions = SliceOptions()
        sliceOptions.Init(buf, pos)
        return cls.InitFromObj(sliceOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, sliceOptions):
        x = SliceOptionsT()
        x._UnPack(sliceOptions)
        return x

    def _UnPack(self, sliceOptions):
        if sliceOptions is None:
            return

    def Pack(self, builder):
        SliceOptionsStart(builder)
        sliceOptions = SliceOptionsEnd(builder)
        return sliceOptions