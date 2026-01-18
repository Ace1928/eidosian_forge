import flatbuffers
from flatbuffers.compat import import_numpy
def Conv3DOptionsAddDilationDFactor(builder, dilationDFactor):
    builder.PrependInt32Slot(5, dilationDFactor, 1)