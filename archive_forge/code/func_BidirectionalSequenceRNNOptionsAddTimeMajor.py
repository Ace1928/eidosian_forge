import flatbuffers
from flatbuffers.compat import import_numpy
def BidirectionalSequenceRNNOptionsAddTimeMajor(builder, timeMajor):
    builder.PrependBoolSlot(0, timeMajor, 0)