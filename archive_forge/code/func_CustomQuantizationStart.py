import flatbuffers
from flatbuffers.compat import import_numpy
def CustomQuantizationStart(builder):
    builder.StartObject(1)