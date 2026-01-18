import flatbuffers
from flatbuffers.compat import import_numpy
class ReverseSequenceOptionsT(object):

    def __init__(self):
        self.seqDim = 0
        self.batchDim = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        reverseSequenceOptions = ReverseSequenceOptions()
        reverseSequenceOptions.Init(buf, pos)
        return cls.InitFromObj(reverseSequenceOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, reverseSequenceOptions):
        x = ReverseSequenceOptionsT()
        x._UnPack(reverseSequenceOptions)
        return x

    def _UnPack(self, reverseSequenceOptions):
        if reverseSequenceOptions is None:
            return
        self.seqDim = reverseSequenceOptions.SeqDim()
        self.batchDim = reverseSequenceOptions.BatchDim()

    def Pack(self, builder):
        ReverseSequenceOptionsStart(builder)
        ReverseSequenceOptionsAddSeqDim(builder, self.seqDim)
        ReverseSequenceOptionsAddBatchDim(builder, self.batchDim)
        reverseSequenceOptions = ReverseSequenceOptionsEnd(builder)
        return reverseSequenceOptions