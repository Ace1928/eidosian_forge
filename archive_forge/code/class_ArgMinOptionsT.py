import flatbuffers
from flatbuffers.compat import import_numpy
class ArgMinOptionsT(object):

    def __init__(self):
        self.outputType = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        argMinOptions = ArgMinOptions()
        argMinOptions.Init(buf, pos)
        return cls.InitFromObj(argMinOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, argMinOptions):
        x = ArgMinOptionsT()
        x._UnPack(argMinOptions)
        return x

    def _UnPack(self, argMinOptions):
        if argMinOptions is None:
            return
        self.outputType = argMinOptions.OutputType()

    def Pack(self, builder):
        ArgMinOptionsStart(builder)
        ArgMinOptionsAddOutputType(builder, self.outputType)
        argMinOptions = ArgMinOptionsEnd(builder)
        return argMinOptions