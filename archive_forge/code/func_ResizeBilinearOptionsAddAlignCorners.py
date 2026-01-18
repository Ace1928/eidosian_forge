import flatbuffers
from flatbuffers.compat import import_numpy
def ResizeBilinearOptionsAddAlignCorners(builder, alignCorners):
    builder.PrependBoolSlot(2, alignCorners, 0)