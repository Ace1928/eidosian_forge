import flatbuffers
from flatbuffers.compat import import_numpy
def Intermediates(self, j):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
    if o != 0:
        a = self._tab.Vector(o)
        return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
    return 0