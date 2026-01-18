import flatbuffers
from flatbuffers.compat import import_numpy
def BucketizeOptionsAddBoundaries(builder, boundaries):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(boundaries), 0)