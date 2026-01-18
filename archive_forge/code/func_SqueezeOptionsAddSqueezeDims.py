import flatbuffers
from flatbuffers.compat import import_numpy
def SqueezeOptionsAddSqueezeDims(builder, squeezeDims):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(squeezeDims), 0)