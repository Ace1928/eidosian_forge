import flatbuffers
from flatbuffers.compat import import_numpy
def UnidirectionalSequenceLSTMOptionsAddCellClip(builder, cellClip):
    builder.PrependFloat32Slot(1, cellClip, 0.0)