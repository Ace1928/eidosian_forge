import flatbuffers
from flatbuffers.compat import import_numpy
class LogSoftmaxOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        logSoftmaxOptions = LogSoftmaxOptions()
        logSoftmaxOptions.Init(buf, pos)
        return cls.InitFromObj(logSoftmaxOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, logSoftmaxOptions):
        x = LogSoftmaxOptionsT()
        x._UnPack(logSoftmaxOptions)
        return x

    def _UnPack(self, logSoftmaxOptions):
        if logSoftmaxOptions is None:
            return

    def Pack(self, builder):
        LogSoftmaxOptionsStart(builder)
        logSoftmaxOptions = LogSoftmaxOptionsEnd(builder)
        return logSoftmaxOptions