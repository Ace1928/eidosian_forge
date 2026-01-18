import flatbuffers
from flatbuffers.compat import import_numpy
def ArgMaxOptionsAddOutputType(builder, outputType):
    builder.PrependInt8Slot(0, outputType, 0)