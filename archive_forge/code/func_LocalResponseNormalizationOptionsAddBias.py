import flatbuffers
from flatbuffers.compat import import_numpy
def LocalResponseNormalizationOptionsAddBias(builder, bias):
    builder.PrependFloat32Slot(1, bias, 0.0)