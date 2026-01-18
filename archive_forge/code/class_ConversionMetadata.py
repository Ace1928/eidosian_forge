import flatbuffers
from flatbuffers.compat import import_numpy
class ConversionMetadata(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ConversionMetadata()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsConversionMetadata(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    def Environment(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = Environment()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Options(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = ConversionOptions()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None