import flatbuffers
from flatbuffers.compat import import_numpy
class RandomOptionsT(object):

    def __init__(self):
        self.seed = 0
        self.seed2 = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        randomOptions = RandomOptions()
        randomOptions.Init(buf, pos)
        return cls.InitFromObj(randomOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, randomOptions):
        x = RandomOptionsT()
        x._UnPack(randomOptions)
        return x

    def _UnPack(self, randomOptions):
        if randomOptions is None:
            return
        self.seed = randomOptions.Seed()
        self.seed2 = randomOptions.Seed2()

    def Pack(self, builder):
        RandomOptionsStart(builder)
        RandomOptionsAddSeed(builder, self.seed)
        RandomOptionsAddSeed2(builder, self.seed2)
        randomOptions = RandomOptionsEnd(builder)
        return randomOptions