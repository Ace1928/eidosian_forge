import flatbuffers
from flatbuffers.compat import import_numpy
def Conv3DOptionsAddStrideW(builder, strideW):
    builder.PrependInt32Slot(2, strideW, 0)