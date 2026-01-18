import flatbuffers
from flatbuffers.compat import import_numpy
def BlockMapIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
    return o == 0