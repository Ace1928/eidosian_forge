import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorAddLargeCustomOptionsOffset(builder, largeCustomOptionsOffset):
    builder.PrependUint64Slot(9, largeCustomOptionsOffset, 0)