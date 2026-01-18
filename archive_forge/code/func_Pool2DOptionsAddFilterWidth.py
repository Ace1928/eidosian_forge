import flatbuffers
from flatbuffers.compat import import_numpy
def Pool2DOptionsAddFilterWidth(builder, filterWidth):
    builder.PrependInt32Slot(3, filterWidth, 0)