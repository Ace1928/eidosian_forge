import flatbuffers
from flatbuffers.compat import import_numpy
def ReverseSequenceOptionsAddSeqDim(builder, seqDim):
    builder.PrependInt32Slot(0, seqDim, 0)