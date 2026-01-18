import flatbuffers
from flatbuffers.compat import import_numpy
def BatchMatMulOptionsAddAdjY(builder, adjY):
    builder.PrependBoolSlot(1, adjY, 0)