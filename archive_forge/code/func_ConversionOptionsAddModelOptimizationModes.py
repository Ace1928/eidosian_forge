import flatbuffers
from flatbuffers.compat import import_numpy
def ConversionOptionsAddModelOptimizationModes(builder, modelOptimizationModes):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(modelOptimizationModes), 0)