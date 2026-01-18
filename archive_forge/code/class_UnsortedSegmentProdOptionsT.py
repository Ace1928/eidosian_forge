import flatbuffers
from flatbuffers.compat import import_numpy
class UnsortedSegmentProdOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        unsortedSegmentProdOptions = UnsortedSegmentProdOptions()
        unsortedSegmentProdOptions.Init(buf, pos)
        return cls.InitFromObj(unsortedSegmentProdOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, unsortedSegmentProdOptions):
        x = UnsortedSegmentProdOptionsT()
        x._UnPack(unsortedSegmentProdOptions)
        return x

    def _UnPack(self, unsortedSegmentProdOptions):
        if unsortedSegmentProdOptions is None:
            return

    def Pack(self, builder):
        UnsortedSegmentProdOptionsStart(builder)
        unsortedSegmentProdOptions = UnsortedSegmentProdOptionsEnd(builder)
        return unsortedSegmentProdOptions