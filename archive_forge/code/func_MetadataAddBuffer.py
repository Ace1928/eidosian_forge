import flatbuffers
from flatbuffers.compat import import_numpy
def MetadataAddBuffer(builder, buffer):
    builder.PrependUint32Slot(1, buffer, 0)