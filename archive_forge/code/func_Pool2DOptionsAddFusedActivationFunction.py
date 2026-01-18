import flatbuffers
from flatbuffers.compat import import_numpy
def Pool2DOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(5, fusedActivationFunction, 0)