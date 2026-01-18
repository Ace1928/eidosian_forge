import flatbuffers
from flatbuffers.compat import import_numpy
class UnsortedSegmentMinOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        unsortedSegmentMinOptions = UnsortedSegmentMinOptions()
        unsortedSegmentMinOptions.Init(buf, pos)
        return cls.InitFromObj(unsortedSegmentMinOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, unsortedSegmentMinOptions):
        x = UnsortedSegmentMinOptionsT()
        x._UnPack(unsortedSegmentMinOptions)
        return x

    def _UnPack(self, unsortedSegmentMinOptions):
        if unsortedSegmentMinOptions is None:
            return

    def Pack(self, builder):
        UnsortedSegmentMinOptionsStart(builder)
        unsortedSegmentMinOptions = UnsortedSegmentMinOptionsEnd(builder)
        return unsortedSegmentMinOptions