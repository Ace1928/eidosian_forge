import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorAddCustomOptions(builder, customOptions):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(customOptions), 0)