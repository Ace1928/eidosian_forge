import flatbuffers
from flatbuffers.compat import import_numpy
class ResizeBilinearOptionsT(object):

    def __init__(self):
        self.alignCorners = False
        self.halfPixelCenters = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        resizeBilinearOptions = ResizeBilinearOptions()
        resizeBilinearOptions.Init(buf, pos)
        return cls.InitFromObj(resizeBilinearOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, resizeBilinearOptions):
        x = ResizeBilinearOptionsT()
        x._UnPack(resizeBilinearOptions)
        return x

    def _UnPack(self, resizeBilinearOptions):
        if resizeBilinearOptions is None:
            return
        self.alignCorners = resizeBilinearOptions.AlignCorners()
        self.halfPixelCenters = resizeBilinearOptions.HalfPixelCenters()

    def Pack(self, builder):
        ResizeBilinearOptionsStart(builder)
        ResizeBilinearOptionsAddAlignCorners(builder, self.alignCorners)
        ResizeBilinearOptionsAddHalfPixelCenters(builder, self.halfPixelCenters)
        resizeBilinearOptions = ResizeBilinearOptionsEnd(builder)
        return resizeBilinearOptions