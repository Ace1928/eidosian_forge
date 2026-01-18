import flatbuffers
from flatbuffers.compat import import_numpy
def SignatureDefAddOutputs(builder, outputs):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0)