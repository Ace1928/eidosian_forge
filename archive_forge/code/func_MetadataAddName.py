import flatbuffers
from flatbuffers.compat import import_numpy
def MetadataAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)