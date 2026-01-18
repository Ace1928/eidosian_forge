import flatbuffers
from flatbuffers.compat import import_numpy
def StridedSliceOptionsAddBeginMask(builder, beginMask):
    builder.PrependInt32Slot(0, beginMask, 0)