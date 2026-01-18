import flatbuffers
from flatbuffers.compat import import_numpy
class Uint8VectorT(object):

    def __init__(self):
        self.values = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        uint8Vector = Uint8Vector()
        uint8Vector.Init(buf, pos)
        return cls.InitFromObj(uint8Vector)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, uint8Vector):
        x = Uint8VectorT()
        x._UnPack(uint8Vector)
        return x

    def _UnPack(self, uint8Vector):
        if uint8Vector is None:
            return
        if not uint8Vector.ValuesIsNone():
            if np is None:
                self.values = []
                for i in range(uint8Vector.ValuesLength()):
                    self.values.append(uint8Vector.Values(i))
            else:
                self.values = uint8Vector.ValuesAsNumpy()

    def Pack(self, builder):
        if self.values is not None:
            if np is not None and type(self.values) is np.ndarray:
                values = builder.CreateNumpyVector(self.values)
            else:
                Uint8VectorStartValuesVector(builder, len(self.values))
                for i in reversed(range(len(self.values))):
                    builder.PrependUint8(self.values[i])
                values = builder.EndVector()
        Uint8VectorStart(builder)
        if self.values is not None:
            Uint8VectorAddValues(builder, values)
        uint8Vector = Uint8VectorEnd(builder)
        return uint8Vector