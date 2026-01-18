import flatbuffers
from flatbuffers.compat import import_numpy
def SqueezeDimsIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
    return o == 0