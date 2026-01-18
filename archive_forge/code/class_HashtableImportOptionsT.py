import flatbuffers
from flatbuffers.compat import import_numpy
class HashtableImportOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        hashtableImportOptions = HashtableImportOptions()
        hashtableImportOptions.Init(buf, pos)
        return cls.InitFromObj(hashtableImportOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, hashtableImportOptions):
        x = HashtableImportOptionsT()
        x._UnPack(hashtableImportOptions)
        return x

    def _UnPack(self, hashtableImportOptions):
        if hashtableImportOptions is None:
            return

    def Pack(self, builder):
        HashtableImportOptionsStart(builder)
        hashtableImportOptions = HashtableImportOptionsEnd(builder)
        return hashtableImportOptions