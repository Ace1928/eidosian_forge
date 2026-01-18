import flatbuffers
from flatbuffers.compat import import_numpy
class SignOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        signOptions = SignOptions()
        signOptions.Init(buf, pos)
        return cls.InitFromObj(signOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, signOptions):
        x = SignOptionsT()
        x._UnPack(signOptions)
        return x

    def _UnPack(self, signOptions):
        if signOptions is None:
            return

    def Pack(self, builder):
        SignOptionsStart(builder)
        signOptions = SignOptionsEnd(builder)
        return signOptions