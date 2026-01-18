import flatbuffers
from flatbuffers.compat import import_numpy
def SparseToDenseOptionsAddValidateIndices(builder, validateIndices):
    builder.PrependBoolSlot(0, validateIndices, 0)