import flatbuffers
from flatbuffers.compat import import_numpy
class SelectOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        selectOptions = SelectOptions()
        selectOptions.Init(buf, pos)
        return cls.InitFromObj(selectOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, selectOptions):
        x = SelectOptionsT()
        x._UnPack(selectOptions)
        return x

    def _UnPack(self, selectOptions):
        if selectOptions is None:
            return

    def Pack(self, builder):
        SelectOptionsStart(builder)
        selectOptions = SelectOptionsEnd(builder)
        return selectOptions