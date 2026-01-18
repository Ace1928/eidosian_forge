import flatbuffers
from flatbuffers.compat import import_numpy
class EqualOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        equalOptions = EqualOptions()
        equalOptions.Init(buf, pos)
        return cls.InitFromObj(equalOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, equalOptions):
        x = EqualOptionsT()
        x._UnPack(equalOptions)
        return x

    def _UnPack(self, equalOptions):
        if equalOptions is None:
            return

    def Pack(self, builder):
        EqualOptionsStart(builder)
        equalOptions = EqualOptionsEnd(builder)
        return equalOptions