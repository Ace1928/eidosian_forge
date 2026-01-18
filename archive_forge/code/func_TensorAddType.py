import flatbuffers
from flatbuffers.compat import import_numpy
def TensorAddType(builder, type):
    builder.PrependInt8Slot(1, type, 0)