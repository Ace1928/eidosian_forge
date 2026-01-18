import flatbuffers
from flatbuffers.compat import import_numpy
def SkipGramOptionsAddMaxSkipSize(builder, maxSkipSize):
    builder.PrependInt32Slot(1, maxSkipSize, 0)