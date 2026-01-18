import flatbuffers
from flatbuffers.compat import import_numpy
def FakeQuantOptionsAddNarrowRange(builder, narrowRange):
    builder.PrependBoolSlot(3, narrowRange, 0)