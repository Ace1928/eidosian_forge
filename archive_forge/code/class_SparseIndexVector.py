import flatbuffers
from flatbuffers.compat import import_numpy
class SparseIndexVector(object):
    NONE = 0
    Int32Vector = 1
    Uint16Vector = 2
    Uint8Vector = 3