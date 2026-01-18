import flatbuffers
from flatbuffers.compat import import_numpy
def QuantizationParametersAddMax(builder, max):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(max), 0)