import flatbuffers
from flatbuffers.compat import import_numpy
def Init(self, buf, pos):
    self._tab = flatbuffers.table.Table(buf, pos)