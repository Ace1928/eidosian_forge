import flatbuffers
from flatbuffers.compat import import_numpy
def SubGraphAddTensors(builder, tensors):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(tensors), 0)