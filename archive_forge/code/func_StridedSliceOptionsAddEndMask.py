import flatbuffers
from flatbuffers.compat import import_numpy
def StridedSliceOptionsAddEndMask(builder, endMask):
    builder.PrependInt32Slot(1, endMask, 0)