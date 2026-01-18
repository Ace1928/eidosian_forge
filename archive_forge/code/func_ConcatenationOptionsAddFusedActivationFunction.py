import flatbuffers
from flatbuffers.compat import import_numpy
def ConcatenationOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(1, fusedActivationFunction, 0)