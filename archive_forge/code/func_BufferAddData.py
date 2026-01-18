import flatbuffers
from flatbuffers.compat import import_numpy
def BufferAddData(builder, data):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(data), 0)