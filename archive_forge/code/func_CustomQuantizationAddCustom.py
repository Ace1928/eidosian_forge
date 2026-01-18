import flatbuffers
from flatbuffers.compat import import_numpy
def CustomQuantizationAddCustom(builder, custom):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(custom), 0)