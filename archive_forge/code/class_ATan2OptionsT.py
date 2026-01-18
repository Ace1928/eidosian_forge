import flatbuffers
from flatbuffers.compat import import_numpy
class ATan2OptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        atan2Options = ATan2Options()
        atan2Options.Init(buf, pos)
        return cls.InitFromObj(atan2Options)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, atan2Options):
        x = ATan2OptionsT()
        x._UnPack(atan2Options)
        return x

    def _UnPack(self, atan2Options):
        if atan2Options is None:
            return

    def Pack(self, builder):
        ATan2OptionsStart(builder)
        atan2Options = ATan2OptionsEnd(builder)
        return atan2Options