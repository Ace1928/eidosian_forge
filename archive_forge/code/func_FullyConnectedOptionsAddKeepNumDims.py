import flatbuffers
from flatbuffers.compat import import_numpy
def FullyConnectedOptionsAddKeepNumDims(builder, keepNumDims):
    builder.PrependBoolSlot(2, keepNumDims, 0)