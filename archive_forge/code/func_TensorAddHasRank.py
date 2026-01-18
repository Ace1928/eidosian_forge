import flatbuffers
from flatbuffers.compat import import_numpy
def TensorAddHasRank(builder, hasRank):
    builder.PrependBoolSlot(8, hasRank, 0)