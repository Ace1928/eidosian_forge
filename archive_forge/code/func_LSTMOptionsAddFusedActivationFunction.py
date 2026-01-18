import flatbuffers
from flatbuffers.compat import import_numpy
def LSTMOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)