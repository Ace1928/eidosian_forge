import flatbuffers
from flatbuffers.compat import import_numpy
def ModelAddBuffers(builder, buffers):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(buffers), 0)