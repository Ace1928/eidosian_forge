import flatbuffers
from flatbuffers.compat import import_numpy
def DimensionMetadataAddArraySegments(builder, arraySegments):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(arraySegments), 0)