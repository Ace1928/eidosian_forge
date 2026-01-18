import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorAddLargeCustomOptionsSize(builder, largeCustomOptionsSize):
    builder.PrependUint64Slot(10, largeCustomOptionsSize, 0)