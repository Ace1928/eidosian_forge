import flatbuffers
from flatbuffers.compat import import_numpy
def CastOptionsAddInDataType(builder, inDataType):
    builder.PrependInt8Slot(0, inDataType, 0)