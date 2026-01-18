import flatbuffers
from flatbuffers.compat import import_numpy
class SplitVOptionsT(object):

    def __init__(self):
        self.numSplits = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        splitVoptions = SplitVOptions()
        splitVoptions.Init(buf, pos)
        return cls.InitFromObj(splitVoptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, splitVoptions):
        x = SplitVOptionsT()
        x._UnPack(splitVoptions)
        return x

    def _UnPack(self, splitVoptions):
        if splitVoptions is None:
            return
        self.numSplits = splitVoptions.NumSplits()

    def Pack(self, builder):
        SplitVOptionsStart(builder)
        SplitVOptionsAddNumSplits(builder, self.numSplits)
        splitVoptions = SplitVOptionsEnd(builder)
        return splitVoptions