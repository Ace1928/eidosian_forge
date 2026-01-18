import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorStartOutputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)