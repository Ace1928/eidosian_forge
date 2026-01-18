import flatbuffers
from flatbuffers.compat import import_numpy
def Container(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
    if o != 0:
        return self._tab.String(o + self._tab.Pos)
    return None