import flatbuffers
from flatbuffers.compat import import_numpy
class OneHotOptionsT(object):

    def __init__(self):
        self.axis = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        oneHotOptions = OneHotOptions()
        oneHotOptions.Init(buf, pos)
        return cls.InitFromObj(oneHotOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, oneHotOptions):
        x = OneHotOptionsT()
        x._UnPack(oneHotOptions)
        return x

    def _UnPack(self, oneHotOptions):
        if oneHotOptions is None:
            return
        self.axis = oneHotOptions.Axis()

    def Pack(self, builder):
        OneHotOptionsStart(builder)
        OneHotOptionsAddAxis(builder, self.axis)
        oneHotOptions = OneHotOptionsEnd(builder)
        return oneHotOptions