import flatbuffers
from flatbuffers.compat import import_numpy
def CumsumOptionsAddReverse(builder, reverse):
    builder.PrependBoolSlot(1, reverse, 0)