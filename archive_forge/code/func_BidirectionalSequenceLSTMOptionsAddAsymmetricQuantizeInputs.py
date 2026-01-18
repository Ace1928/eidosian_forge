import flatbuffers
from flatbuffers.compat import import_numpy
def BidirectionalSequenceLSTMOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(5, asymmetricQuantizeInputs, 0)