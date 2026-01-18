import flatbuffers
from flatbuffers.compat import import_numpy
def GeluOptionsAddApproximate(builder, approximate):
    builder.PrependBoolSlot(0, approximate, 0)