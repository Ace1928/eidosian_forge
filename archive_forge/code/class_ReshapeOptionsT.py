import flatbuffers
from flatbuffers.compat import import_numpy
class ReshapeOptionsT(object):

    def __init__(self):
        self.newShape = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        reshapeOptions = ReshapeOptions()
        reshapeOptions.Init(buf, pos)
        return cls.InitFromObj(reshapeOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, reshapeOptions):
        x = ReshapeOptionsT()
        x._UnPack(reshapeOptions)
        return x

    def _UnPack(self, reshapeOptions):
        if reshapeOptions is None:
            return
        if not reshapeOptions.NewShapeIsNone():
            if np is None:
                self.newShape = []
                for i in range(reshapeOptions.NewShapeLength()):
                    self.newShape.append(reshapeOptions.NewShape(i))
            else:
                self.newShape = reshapeOptions.NewShapeAsNumpy()

    def Pack(self, builder):
        if self.newShape is not None:
            if np is not None and type(self.newShape) is np.ndarray:
                newShape = builder.CreateNumpyVector(self.newShape)
            else:
                ReshapeOptionsStartNewShapeVector(builder, len(self.newShape))
                for i in reversed(range(len(self.newShape))):
                    builder.PrependInt32(self.newShape[i])
                newShape = builder.EndVector()
        ReshapeOptionsStart(builder)
        if self.newShape is not None:
            ReshapeOptionsAddNewShape(builder, newShape)
        reshapeOptions = ReshapeOptionsEnd(builder)
        return reshapeOptions