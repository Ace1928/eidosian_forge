import flatbuffers
from flatbuffers.compat import import_numpy
def DimensionMetadataAddArrayIndicesType(builder, arrayIndicesType):
    builder.PrependUint8Slot(4, arrayIndicesType, 0)