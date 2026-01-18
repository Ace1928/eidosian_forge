import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorAddIntermediates(builder, intermediates):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(intermediates), 0)