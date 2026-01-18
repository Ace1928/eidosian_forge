import flatbuffers
from flatbuffers.compat import import_numpy
def FakeQuantOptionsAddNumBits(builder, numBits):
    builder.PrependInt32Slot(2, numBits, 0)