import flatbuffers
from flatbuffers.compat import import_numpy
def CustomOptionsIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
    return o == 0