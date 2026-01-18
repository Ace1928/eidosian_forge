import flatbuffers
from flatbuffers.compat import import_numpy
def SignatureDefsIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
    return o == 0