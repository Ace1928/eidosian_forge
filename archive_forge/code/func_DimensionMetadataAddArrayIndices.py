import flatbuffers
from flatbuffers.compat import import_numpy
def DimensionMetadataAddArrayIndices(builder, arrayIndices):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(arrayIndices), 0)