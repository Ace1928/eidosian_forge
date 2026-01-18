import flatbuffers
from flatbuffers.compat import import_numpy
def AlignCorners(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
    if o != 0:
        return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
    return False