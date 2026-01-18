import flatbuffers
from flatbuffers.compat import import_numpy
def VariantSubTypeAddHasRank(builder, hasRank):
    builder.PrependBoolSlot(2, hasRank, 0)