import flatbuffers
from flatbuffers.compat import import_numpy
class GeluOptionsT(object):

    def __init__(self):
        self.approximate = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        geluOptions = GeluOptions()
        geluOptions.Init(buf, pos)
        return cls.InitFromObj(geluOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, geluOptions):
        x = GeluOptionsT()
        x._UnPack(geluOptions)
        return x

    def _UnPack(self, geluOptions):
        if geluOptions is None:
            return
        self.approximate = geluOptions.Approximate()

    def Pack(self, builder):
        GeluOptionsStart(builder)
        GeluOptionsAddApproximate(builder, self.approximate)
        geluOptions = GeluOptionsEnd(builder)
        return geluOptions