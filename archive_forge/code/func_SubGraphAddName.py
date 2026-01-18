import flatbuffers
from flatbuffers.compat import import_numpy
def SubGraphAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)