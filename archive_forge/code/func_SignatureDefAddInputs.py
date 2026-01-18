import flatbuffers
from flatbuffers.compat import import_numpy
def SignatureDefAddInputs(builder, inputs):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0)