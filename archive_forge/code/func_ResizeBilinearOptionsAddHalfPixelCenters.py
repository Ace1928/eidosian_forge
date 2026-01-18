import flatbuffers
from flatbuffers.compat import import_numpy
def ResizeBilinearOptionsAddHalfPixelCenters(builder, halfPixelCenters):
    builder.PrependBoolSlot(3, halfPixelCenters, 0)