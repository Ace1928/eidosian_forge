import flatbuffers
from flatbuffers.compat import import_numpy
class NotEqualOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        notEqualOptions = NotEqualOptions()
        notEqualOptions.Init(buf, pos)
        return cls.InitFromObj(notEqualOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, notEqualOptions):
        x = NotEqualOptionsT()
        x._UnPack(notEqualOptions)
        return x

    def _UnPack(self, notEqualOptions):
        if notEqualOptions is None:
            return

    def Pack(self, builder):
        NotEqualOptionsStart(builder)
        notEqualOptions = NotEqualOptionsEnd(builder)
        return notEqualOptions