import flatbuffers
from flatbuffers.compat import import_numpy
def RandomOptionsAddSeed2(builder, seed2):
    builder.PrependInt64Slot(1, seed2, 0)