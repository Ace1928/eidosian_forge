import flatbuffers
from flatbuffers.compat import import_numpy
def LSTMOptionsAddKernelType(builder, kernelType):
    builder.PrependInt8Slot(3, kernelType, 0)