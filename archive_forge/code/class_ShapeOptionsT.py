import flatbuffers
from flatbuffers.compat import import_numpy
class ShapeOptionsT(object):

    def __init__(self):
        self.outType = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        shapeOptions = ShapeOptions()
        shapeOptions.Init(buf, pos)
        return cls.InitFromObj(shapeOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, shapeOptions):
        x = ShapeOptionsT()
        x._UnPack(shapeOptions)
        return x

    def _UnPack(self, shapeOptions):
        if shapeOptions is None:
            return
        self.outType = shapeOptions.OutType()

    def Pack(self, builder):
        ShapeOptionsStart(builder)
        ShapeOptionsAddOutType(builder, self.outType)
        shapeOptions = ShapeOptionsEnd(builder)
        return shapeOptions