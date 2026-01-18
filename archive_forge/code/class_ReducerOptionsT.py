import flatbuffers
from flatbuffers.compat import import_numpy
class ReducerOptionsT(object):

    def __init__(self):
        self.keepDims = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        reducerOptions = ReducerOptions()
        reducerOptions.Init(buf, pos)
        return cls.InitFromObj(reducerOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, reducerOptions):
        x = ReducerOptionsT()
        x._UnPack(reducerOptions)
        return x

    def _UnPack(self, reducerOptions):
        if reducerOptions is None:
            return
        self.keepDims = reducerOptions.KeepDims()

    def Pack(self, builder):
        ReducerOptionsStart(builder)
        ReducerOptionsAddKeepDims(builder, self.keepDims)
        reducerOptions = ReducerOptionsEnd(builder)
        return reducerOptions