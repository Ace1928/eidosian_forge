import flatbuffers
from flatbuffers.compat import import_numpy
def ResizeNearestNeighborOptionsAddHalfPixelCenters(builder, halfPixelCenters):
    builder.PrependBoolSlot(1, halfPixelCenters, 0)