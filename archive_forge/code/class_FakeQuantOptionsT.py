import flatbuffers
from flatbuffers.compat import import_numpy
class FakeQuantOptionsT(object):

    def __init__(self):
        self.min = 0.0
        self.max = 0.0
        self.numBits = 0
        self.narrowRange = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        fakeQuantOptions = FakeQuantOptions()
        fakeQuantOptions.Init(buf, pos)
        return cls.InitFromObj(fakeQuantOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, fakeQuantOptions):
        x = FakeQuantOptionsT()
        x._UnPack(fakeQuantOptions)
        return x

    def _UnPack(self, fakeQuantOptions):
        if fakeQuantOptions is None:
            return
        self.min = fakeQuantOptions.Min()
        self.max = fakeQuantOptions.Max()
        self.numBits = fakeQuantOptions.NumBits()
        self.narrowRange = fakeQuantOptions.NarrowRange()

    def Pack(self, builder):
        FakeQuantOptionsStart(builder)
        FakeQuantOptionsAddMin(builder, self.min)
        FakeQuantOptionsAddMax(builder, self.max)
        FakeQuantOptionsAddNumBits(builder, self.numBits)
        FakeQuantOptionsAddNarrowRange(builder, self.narrowRange)
        fakeQuantOptions = FakeQuantOptionsEnd(builder)
        return fakeQuantOptions