import flatbuffers
from flatbuffers.compat import import_numpy
def TensorAddShape(builder, shape):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(shape), 0)