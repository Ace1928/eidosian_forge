import flatbuffers
from flatbuffers.compat import import_numpy
def Conv3DOptionsAddDilationHFactor(builder, dilationHFactor):
    builder.PrependInt32Slot(7, dilationHFactor, 1)