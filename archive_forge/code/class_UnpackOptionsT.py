import flatbuffers
from flatbuffers.compat import import_numpy
class UnpackOptionsT(object):

    def __init__(self):
        self.num = 0
        self.axis = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        unpackOptions = UnpackOptions()
        unpackOptions.Init(buf, pos)
        return cls.InitFromObj(unpackOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, unpackOptions):
        x = UnpackOptionsT()
        x._UnPack(unpackOptions)
        return x

    def _UnPack(self, unpackOptions):
        if unpackOptions is None:
            return
        self.num = unpackOptions.Num()
        self.axis = unpackOptions.Axis()

    def Pack(self, builder):
        UnpackOptionsStart(builder)
        UnpackOptionsAddNum(builder, self.num)
        UnpackOptionsAddAxis(builder, self.axis)
        unpackOptions = UnpackOptionsEnd(builder)
        return unpackOptions