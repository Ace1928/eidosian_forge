import flatbuffers
from flatbuffers.compat import import_numpy
class HashtableOptionsT(object):

    def __init__(self):
        self.tableId = 0
        self.keyDtype = 0
        self.valueDtype = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        hashtableOptions = HashtableOptions()
        hashtableOptions.Init(buf, pos)
        return cls.InitFromObj(hashtableOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, hashtableOptions):
        x = HashtableOptionsT()
        x._UnPack(hashtableOptions)
        return x

    def _UnPack(self, hashtableOptions):
        if hashtableOptions is None:
            return
        self.tableId = hashtableOptions.TableId()
        self.keyDtype = hashtableOptions.KeyDtype()
        self.valueDtype = hashtableOptions.ValueDtype()

    def Pack(self, builder):
        HashtableOptionsStart(builder)
        HashtableOptionsAddTableId(builder, self.tableId)
        HashtableOptionsAddKeyDtype(builder, self.keyDtype)
        HashtableOptionsAddValueDtype(builder, self.valueDtype)
        hashtableOptions = HashtableOptionsEnd(builder)
        return hashtableOptions