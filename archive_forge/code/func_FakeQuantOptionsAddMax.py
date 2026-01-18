import flatbuffers
from flatbuffers.compat import import_numpy
def FakeQuantOptionsAddMax(builder, max):
    builder.PrependFloat32Slot(1, max, 0.0)