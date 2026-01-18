import flatbuffers
from flatbuffers.compat import import_numpy
def BufferAddSize(builder, size):
    builder.PrependUint64Slot(2, size, 0)