import flatbuffers
from flatbuffers.compat import import_numpy
def StridedSliceOptionsAddShrinkAxisMask(builder, shrinkAxisMask):
    builder.PrependInt32Slot(4, shrinkAxisMask, 0)