import flatbuffers
from flatbuffers.compat import import_numpy
def ConversionOptionsAddAllowCustomOps(builder, allowCustomOps):
    builder.PrependBoolSlot(1, allowCustomOps, 0)