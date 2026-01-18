import flatbuffers
from flatbuffers.compat import import_numpy
def ReducerOptionsAddKeepDims(builder, keepDims):
    builder.PrependBoolSlot(0, keepDims, 0)