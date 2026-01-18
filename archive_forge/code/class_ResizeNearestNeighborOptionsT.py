import flatbuffers
from flatbuffers.compat import import_numpy
class ResizeNearestNeighborOptionsT(object):

    def __init__(self):
        self.alignCorners = False
        self.halfPixelCenters = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        resizeNearestNeighborOptions = ResizeNearestNeighborOptions()
        resizeNearestNeighborOptions.Init(buf, pos)
        return cls.InitFromObj(resizeNearestNeighborOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, resizeNearestNeighborOptions):
        x = ResizeNearestNeighborOptionsT()
        x._UnPack(resizeNearestNeighborOptions)
        return x

    def _UnPack(self, resizeNearestNeighborOptions):
        if resizeNearestNeighborOptions is None:
            return
        self.alignCorners = resizeNearestNeighborOptions.AlignCorners()
        self.halfPixelCenters = resizeNearestNeighborOptions.HalfPixelCenters()

    def Pack(self, builder):
        ResizeNearestNeighborOptionsStart(builder)
        ResizeNearestNeighborOptionsAddAlignCorners(builder, self.alignCorners)
        ResizeNearestNeighborOptionsAddHalfPixelCenters(builder, self.halfPixelCenters)
        resizeNearestNeighborOptions = ResizeNearestNeighborOptionsEnd(builder)
        return resizeNearestNeighborOptions