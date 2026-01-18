import flatbuffers
from flatbuffers.compat import import_numpy
def EmbeddingLookupSparseOptionsAddCombiner(builder, combiner):
    builder.PrependInt8Slot(0, combiner, 0)