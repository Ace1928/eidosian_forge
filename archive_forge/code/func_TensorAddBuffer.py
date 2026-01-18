import flatbuffers
from flatbuffers.compat import import_numpy
def TensorAddBuffer(builder, buffer):
    builder.PrependUint32Slot(2, buffer, 0)