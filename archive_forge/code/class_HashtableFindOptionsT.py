import flatbuffers
from flatbuffers.compat import import_numpy
class HashtableFindOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        hashtableFindOptions = HashtableFindOptions()
        hashtableFindOptions.Init(buf, pos)
        return cls.InitFromObj(hashtableFindOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, hashtableFindOptions):
        x = HashtableFindOptionsT()
        x._UnPack(hashtableFindOptions)
        return x

    def _UnPack(self, hashtableFindOptions):
        if hashtableFindOptions is None:
            return

    def Pack(self, builder):
        HashtableFindOptionsStart(builder)
        hashtableFindOptions = HashtableFindOptionsEnd(builder)
        return hashtableFindOptions