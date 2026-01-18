import flatbuffers
from flatbuffers.compat import import_numpy
def LocalResponseNormalizationOptionsAddRadius(builder, radius):
    builder.PrependInt32Slot(0, radius, 0)