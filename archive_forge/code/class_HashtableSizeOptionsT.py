import flatbuffers
from flatbuffers.compat import import_numpy
class HashtableSizeOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        hashtableSizeOptions = HashtableSizeOptions()
        hashtableSizeOptions.Init(buf, pos)
        return cls.InitFromObj(hashtableSizeOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, hashtableSizeOptions):
        x = HashtableSizeOptionsT()
        x._UnPack(hashtableSizeOptions)
        return x

    def _UnPack(self, hashtableSizeOptions):
        if hashtableSizeOptions is None:
            return

    def Pack(self, builder):
        HashtableSizeOptionsStart(builder)
        hashtableSizeOptions = HashtableSizeOptionsEnd(builder)
        return hashtableSizeOptions