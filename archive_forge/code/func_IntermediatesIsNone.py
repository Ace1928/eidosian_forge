import flatbuffers
from flatbuffers.compat import import_numpy
def IntermediatesIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
    return o == 0