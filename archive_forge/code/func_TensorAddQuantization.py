import flatbuffers
from flatbuffers.compat import import_numpy
def TensorAddQuantization(builder, quantization):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(quantization), 0)