import flatbuffers
from flatbuffers.compat import import_numpy
def StridedSliceOptionsAddEllipsisMask(builder, ellipsisMask):
    builder.PrependInt32Slot(2, ellipsisMask, 0)