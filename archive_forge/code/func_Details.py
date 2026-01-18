import flatbuffers
from flatbuffers.compat import import_numpy
def Details(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
    if o != 0:
        from flatbuffers.table import Table
        obj = Table(bytearray(), 0)
        self._tab.Union(obj, o)
        return obj
    return None