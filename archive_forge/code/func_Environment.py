import flatbuffers
from flatbuffers.compat import import_numpy
def Environment(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
    if o != 0:
        x = self._tab.Indirect(o + self._tab.Pos)
        obj = Environment()
        obj.Init(self._tab.Bytes, x)
        return obj
    return None