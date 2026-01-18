import flatbuffers
from flatbuffers.compat import import_numpy
def CastOptionsAddOutDataType(builder, outDataType):
    builder.PrependInt8Slot(1, outDataType, 0)