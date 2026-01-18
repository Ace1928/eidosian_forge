import flatbuffers
from flatbuffers.compat import import_numpy
def TensorStart(builder):
    builder.StartObject(10)