import flatbuffers
from flatbuffers.compat import import_numpy
def ReshapeOptionsAddNewShape(builder, newShape):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(newShape), 0)