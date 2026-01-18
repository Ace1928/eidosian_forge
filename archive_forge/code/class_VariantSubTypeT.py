import flatbuffers
from flatbuffers.compat import import_numpy
class VariantSubTypeT(object):

    def __init__(self):
        self.shape = None
        self.type = 0
        self.hasRank = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        variantSubType = VariantSubType()
        variantSubType.Init(buf, pos)
        return cls.InitFromObj(variantSubType)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, variantSubType):
        x = VariantSubTypeT()
        x._UnPack(variantSubType)
        return x

    def _UnPack(self, variantSubType):
        if variantSubType is None:
            return
        if not variantSubType.ShapeIsNone():
            if np is None:
                self.shape = []
                for i in range(variantSubType.ShapeLength()):
                    self.shape.append(variantSubType.Shape(i))
            else:
                self.shape = variantSubType.ShapeAsNumpy()
        self.type = variantSubType.Type()
        self.hasRank = variantSubType.HasRank()

    def Pack(self, builder):
        if self.shape is not None:
            if np is not None and type(self.shape) is np.ndarray:
                shape = builder.CreateNumpyVector(self.shape)
            else:
                VariantSubTypeStartShapeVector(builder, len(self.shape))
                for i in reversed(range(len(self.shape))):
                    builder.PrependInt32(self.shape[i])
                shape = builder.EndVector()
        VariantSubTypeStart(builder)
        if self.shape is not None:
            VariantSubTypeAddShape(builder, shape)
        VariantSubTypeAddType(builder, self.type)
        VariantSubTypeAddHasRank(builder, self.hasRank)
        variantSubType = VariantSubTypeEnd(builder)
        return variantSubType