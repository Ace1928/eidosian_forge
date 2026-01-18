import flatbuffers
from flatbuffers.compat import import_numpy
def ConversionOptionsAddForceSelectTfOps(builder, forceSelectTfOps):
    builder.PrependBoolSlot(3, forceSelectTfOps, 0)