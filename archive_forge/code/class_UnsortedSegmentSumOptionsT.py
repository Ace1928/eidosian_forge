import flatbuffers
from flatbuffers.compat import import_numpy
class UnsortedSegmentSumOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        unsortedSegmentSumOptions = UnsortedSegmentSumOptions()
        unsortedSegmentSumOptions.Init(buf, pos)
        return cls.InitFromObj(unsortedSegmentSumOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, unsortedSegmentSumOptions):
        x = UnsortedSegmentSumOptionsT()
        x._UnPack(unsortedSegmentSumOptions)
        return x

    def _UnPack(self, unsortedSegmentSumOptions):
        if unsortedSegmentSumOptions is None:
            return

    def Pack(self, builder):
        UnsortedSegmentSumOptionsStart(builder)
        unsortedSegmentSumOptions = UnsortedSegmentSumOptionsEnd(builder)
        return unsortedSegmentSumOptions