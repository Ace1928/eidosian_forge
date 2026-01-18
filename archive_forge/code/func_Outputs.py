import flatbuffers
from flatbuffers.compat import import_numpy
def Outputs(self, j):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
    if o != 0:
        x = self._tab.Vector(o)
        x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
        x = self._tab.Indirect(x)
        obj = TensorMap()
        obj.Init(self._tab.Bytes, x)
        return obj
    return None