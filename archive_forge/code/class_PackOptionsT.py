import flatbuffers
from flatbuffers.compat import import_numpy
class PackOptionsT(object):

    def __init__(self):
        self.valuesCount = 0
        self.axis = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        packOptions = PackOptions()
        packOptions.Init(buf, pos)
        return cls.InitFromObj(packOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, packOptions):
        x = PackOptionsT()
        x._UnPack(packOptions)
        return x

    def _UnPack(self, packOptions):
        if packOptions is None:
            return
        self.valuesCount = packOptions.ValuesCount()
        self.axis = packOptions.Axis()

    def Pack(self, builder):
        PackOptionsStart(builder)
        PackOptionsAddValuesCount(builder, self.valuesCount)
        PackOptionsAddAxis(builder, self.axis)
        packOptions = PackOptionsEnd(builder)
        return packOptions