import flatbuffers
from flatbuffers.compat import import_numpy
def QuantizationParametersAddZeroPoint(builder, zeroPoint):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(zeroPoint), 0)