import flatbuffers
from flatbuffers.compat import import_numpy
class ResizeBilinearOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ResizeBilinearOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsResizeBilinearOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    @classmethod
    def ResizeBilinearOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'TFL3', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    def AlignCorners(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    def HalfPixelCenters(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False