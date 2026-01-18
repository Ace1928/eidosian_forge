import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorCodeAddCustomCode(builder, customCode):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(customCode), 0)