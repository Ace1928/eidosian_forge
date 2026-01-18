import flatbuffers
from flatbuffers.compat import import_numpy
class UnsortedSegmentMaxOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        unsortedSegmentMaxOptions = UnsortedSegmentMaxOptions()
        unsortedSegmentMaxOptions.Init(buf, pos)
        return cls.InitFromObj(unsortedSegmentMaxOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, unsortedSegmentMaxOptions):
        x = UnsortedSegmentMaxOptionsT()
        x._UnPack(unsortedSegmentMaxOptions)
        return x

    def _UnPack(self, unsortedSegmentMaxOptions):
        if unsortedSegmentMaxOptions is None:
            return

    def Pack(self, builder):
        UnsortedSegmentMaxOptionsStart(builder)
        unsortedSegmentMaxOptions = UnsortedSegmentMaxOptionsEnd(builder)
        return unsortedSegmentMaxOptions