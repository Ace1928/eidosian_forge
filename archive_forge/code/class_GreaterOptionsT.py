import flatbuffers
from flatbuffers.compat import import_numpy
class GreaterOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        greaterOptions = GreaterOptions()
        greaterOptions.Init(buf, pos)
        return cls.InitFromObj(greaterOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, greaterOptions):
        x = GreaterOptionsT()
        x._UnPack(greaterOptions)
        return x

    def _UnPack(self, greaterOptions):
        if greaterOptions is None:
            return

    def Pack(self, builder):
        GreaterOptionsStart(builder)
        greaterOptions = GreaterOptionsEnd(builder)
        return greaterOptions