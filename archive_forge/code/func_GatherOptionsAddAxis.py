import flatbuffers
from flatbuffers.compat import import_numpy
def GatherOptionsAddAxis(builder, axis):
    builder.PrependInt32Slot(0, axis, 0)