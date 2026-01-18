import flatbuffers
from flatbuffers.compat import import_numpy
def Uint16VectorStartValuesVector(builder, numElems):
    return builder.StartVector(2, numElems, 2)