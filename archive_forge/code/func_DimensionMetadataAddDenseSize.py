import flatbuffers
from flatbuffers.compat import import_numpy
def DimensionMetadataAddDenseSize(builder, denseSize):
    builder.PrependInt32Slot(1, denseSize, 0)