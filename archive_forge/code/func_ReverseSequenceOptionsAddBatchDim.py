import flatbuffers
from flatbuffers.compat import import_numpy
def ReverseSequenceOptionsAddBatchDim(builder, batchDim):
    builder.PrependInt32Slot(1, batchDim, 0)