import flatbuffers
from flatbuffers.compat import import_numpy
def SqueezeOptionsStartSqueezeDimsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)