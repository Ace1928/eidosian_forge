import flatbuffers
from flatbuffers.compat import import_numpy
def VariantTensorsIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
    return o == 0