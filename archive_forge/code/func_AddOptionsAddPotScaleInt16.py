import flatbuffers
from flatbuffers.compat import import_numpy
def AddOptionsAddPotScaleInt16(builder, potScaleInt16):
    builder.PrependBoolSlot(1, potScaleInt16, 1)