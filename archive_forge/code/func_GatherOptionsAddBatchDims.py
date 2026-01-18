import flatbuffers
from flatbuffers.compat import import_numpy
def GatherOptionsAddBatchDims(builder, batchDims):
    builder.PrependInt32Slot(1, batchDims, 0)