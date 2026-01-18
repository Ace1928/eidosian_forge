import flatbuffers
from flatbuffers.compat import import_numpy
def MetadataIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
    return o == 0