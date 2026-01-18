import flatbuffers
from flatbuffers.compat import import_numpy
class NonMaxSuppressionV5OptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        nonMaxSuppressionV5Options = NonMaxSuppressionV5Options()
        nonMaxSuppressionV5Options.Init(buf, pos)
        return cls.InitFromObj(nonMaxSuppressionV5Options)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, nonMaxSuppressionV5Options):
        x = NonMaxSuppressionV5OptionsT()
        x._UnPack(nonMaxSuppressionV5Options)
        return x

    def _UnPack(self, nonMaxSuppressionV5Options):
        if nonMaxSuppressionV5Options is None:
            return

    def Pack(self, builder):
        NonMaxSuppressionV5OptionsStart(builder)
        nonMaxSuppressionV5Options = NonMaxSuppressionV5OptionsEnd(builder)
        return nonMaxSuppressionV5Options