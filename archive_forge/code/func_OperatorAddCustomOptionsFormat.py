import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorAddCustomOptionsFormat(builder, customOptionsFormat):
    builder.PrependInt8Slot(6, customOptionsFormat, 0)