import flatbuffers
from flatbuffers.compat import import_numpy
class LogicalOrOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        logicalOrOptions = LogicalOrOptions()
        logicalOrOptions.Init(buf, pos)
        return cls.InitFromObj(logicalOrOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, logicalOrOptions):
        x = LogicalOrOptionsT()
        x._UnPack(logicalOrOptions)
        return x

    def _UnPack(self, logicalOrOptions):
        if logicalOrOptions is None:
            return

    def Pack(self, builder):
        LogicalOrOptionsStart(builder)
        logicalOrOptions = LogicalOrOptionsEnd(builder)
        return logicalOrOptions