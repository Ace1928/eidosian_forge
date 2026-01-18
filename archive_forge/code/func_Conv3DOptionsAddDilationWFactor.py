import flatbuffers
from flatbuffers.compat import import_numpy
def Conv3DOptionsAddDilationWFactor(builder, dilationWFactor):
    builder.PrependInt32Slot(6, dilationWFactor, 1)