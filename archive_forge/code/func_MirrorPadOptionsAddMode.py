import flatbuffers
from flatbuffers.compat import import_numpy
def MirrorPadOptionsAddMode(builder, mode):
    builder.PrependInt8Slot(0, mode, 0)