import flatbuffers
from flatbuffers.compat import import_numpy
def IfOptionsAddThenSubgraphIndex(builder, thenSubgraphIndex):
    builder.PrependInt32Slot(0, thenSubgraphIndex, 0)