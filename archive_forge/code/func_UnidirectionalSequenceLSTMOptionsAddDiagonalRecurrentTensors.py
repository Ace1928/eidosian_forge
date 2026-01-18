import flatbuffers
from flatbuffers.compat import import_numpy
def UnidirectionalSequenceLSTMOptionsAddDiagonalRecurrentTensors(builder, diagonalRecurrentTensors):
    builder.PrependBoolSlot(5, diagonalRecurrentTensors, 0)