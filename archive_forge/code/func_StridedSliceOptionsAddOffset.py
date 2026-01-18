import flatbuffers
from flatbuffers.compat import import_numpy
def StridedSliceOptionsAddOffset(builder, offset):
    builder.PrependBoolSlot(5, offset, 0)