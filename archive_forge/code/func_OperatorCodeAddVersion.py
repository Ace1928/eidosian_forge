import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorCodeAddVersion(builder, version):
    builder.PrependInt32Slot(2, version, 1)