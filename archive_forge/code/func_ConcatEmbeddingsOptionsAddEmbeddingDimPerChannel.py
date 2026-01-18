import flatbuffers
from flatbuffers.compat import import_numpy
def ConcatEmbeddingsOptionsAddEmbeddingDimPerChannel(builder, embeddingDimPerChannel):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(embeddingDimPerChannel), 0)