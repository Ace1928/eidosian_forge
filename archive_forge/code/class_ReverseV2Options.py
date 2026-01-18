import flatbuffers
from flatbuffers.compat import import_numpy
class ReverseV2Options(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReverseV2Options()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReverseV2Options(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    @classmethod
    def ReverseV2OptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'TFL3', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)