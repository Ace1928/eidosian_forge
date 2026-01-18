import flatbuffers
from flatbuffers.compat import import_numpy
class StridedSliceOptionsT(object):

    def __init__(self):
        self.beginMask = 0
        self.endMask = 0
        self.ellipsisMask = 0
        self.newAxisMask = 0
        self.shrinkAxisMask = 0
        self.offset = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        stridedSliceOptions = StridedSliceOptions()
        stridedSliceOptions.Init(buf, pos)
        return cls.InitFromObj(stridedSliceOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, stridedSliceOptions):
        x = StridedSliceOptionsT()
        x._UnPack(stridedSliceOptions)
        return x

    def _UnPack(self, stridedSliceOptions):
        if stridedSliceOptions is None:
            return
        self.beginMask = stridedSliceOptions.BeginMask()
        self.endMask = stridedSliceOptions.EndMask()
        self.ellipsisMask = stridedSliceOptions.EllipsisMask()
        self.newAxisMask = stridedSliceOptions.NewAxisMask()
        self.shrinkAxisMask = stridedSliceOptions.ShrinkAxisMask()
        self.offset = stridedSliceOptions.Offset()

    def Pack(self, builder):
        StridedSliceOptionsStart(builder)
        StridedSliceOptionsAddBeginMask(builder, self.beginMask)
        StridedSliceOptionsAddEndMask(builder, self.endMask)
        StridedSliceOptionsAddEllipsisMask(builder, self.ellipsisMask)
        StridedSliceOptionsAddNewAxisMask(builder, self.newAxisMask)
        StridedSliceOptionsAddShrinkAxisMask(builder, self.shrinkAxisMask)
        StridedSliceOptionsAddOffset(builder, self.offset)
        stridedSliceOptions = StridedSliceOptionsEnd(builder)
        return stridedSliceOptions