import flatbuffers
from flatbuffers.compat import import_numpy
def CallOptionsAddSubgraph(builder, subgraph):
    builder.PrependUint32Slot(0, subgraph, 0)