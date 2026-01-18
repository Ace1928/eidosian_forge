import flatbuffers
from flatbuffers.compat import import_numpy
class NonMaxSuppressionV4OptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        nonMaxSuppressionV4Options = NonMaxSuppressionV4Options()
        nonMaxSuppressionV4Options.Init(buf, pos)
        return cls.InitFromObj(nonMaxSuppressionV4Options)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, nonMaxSuppressionV4Options):
        x = NonMaxSuppressionV4OptionsT()
        x._UnPack(nonMaxSuppressionV4Options)
        return x

    def _UnPack(self, nonMaxSuppressionV4Options):
        if nonMaxSuppressionV4Options is None:
            return

    def Pack(self, builder):
        NonMaxSuppressionV4OptionsStart(builder)
        nonMaxSuppressionV4Options = NonMaxSuppressionV4OptionsEnd(builder)
        return nonMaxSuppressionV4Options