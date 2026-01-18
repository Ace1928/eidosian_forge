import flatbuffers
from flatbuffers.compat import import_numpy
def BatchMatMulOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(2, asymmetricQuantizeInputs, 0)