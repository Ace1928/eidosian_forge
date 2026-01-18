import flatbuffers
from flatbuffers.compat import import_numpy
class SegmentSumOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        segmentSumOptions = SegmentSumOptions()
        segmentSumOptions.Init(buf, pos)
        return cls.InitFromObj(segmentSumOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, segmentSumOptions):
        x = SegmentSumOptionsT()
        x._UnPack(segmentSumOptions)
        return x

    def _UnPack(self, segmentSumOptions):
        if segmentSumOptions is None:
            return

    def Pack(self, builder):
        SegmentSumOptionsStart(builder)
        segmentSumOptions = SegmentSumOptionsEnd(builder)
        return segmentSumOptions