import flatbuffers
from flatbuffers.compat import import_numpy
class TensorMapT(object):

    def __init__(self):
        self.name = None
        self.tensorIndex = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        tensorMap = TensorMap()
        tensorMap.Init(buf, pos)
        return cls.InitFromObj(tensorMap)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, tensorMap):
        x = TensorMapT()
        x._UnPack(tensorMap)
        return x

    def _UnPack(self, tensorMap):
        if tensorMap is None:
            return
        self.name = tensorMap.Name()
        self.tensorIndex = tensorMap.TensorIndex()

    def Pack(self, builder):
        if self.name is not None:
            name = builder.CreateString(self.name)
        TensorMapStart(builder)
        if self.name is not None:
            TensorMapAddName(builder, name)
        TensorMapAddTensorIndex(builder, self.tensorIndex)
        tensorMap = TensorMapEnd(builder)
        return tensorMap