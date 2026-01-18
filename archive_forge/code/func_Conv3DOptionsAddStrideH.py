import flatbuffers
from flatbuffers.compat import import_numpy
def Conv3DOptionsAddStrideH(builder, strideH):
    builder.PrependInt32Slot(3, strideH, 0)