import flatbuffers
from flatbuffers.compat import import_numpy
class SparsityBlockSizeT(object):

    def __init__(self):
        self.values = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sparsityBlockSize = SparsityBlockSize()
        sparsityBlockSize.Init(buf, pos)
        return cls.InitFromObj(sparsityBlockSize)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, sparsityBlockSize):
        x = SparsityBlockSizeT()
        x._UnPack(sparsityBlockSize)
        return x

    def _UnPack(self, sparsityBlockSize):
        if sparsityBlockSize is None:
            return
        if not sparsityBlockSize.ValuesIsNone():
            if np is None:
                self.values = []
                for i in range(sparsityBlockSize.ValuesLength()):
                    self.values.append(sparsityBlockSize.Values(i))
            else:
                self.values = sparsityBlockSize.ValuesAsNumpy()

    def Pack(self, builder):
        if self.values is not None:
            if np is not None and type(self.values) is np.ndarray:
                values = builder.CreateNumpyVector(self.values)
            else:
                SparsityBlockSizeStartValuesVector(builder, len(self.values))
                for i in reversed(range(len(self.values))):
                    builder.PrependUint32(self.values[i])
                values = builder.EndVector()
        SparsityBlockSizeStart(builder)
        if self.values is not None:
            SparsityBlockSizeAddValues(builder, values)
        sparsityBlockSize = SparsityBlockSizeEnd(builder)
        return sparsityBlockSize