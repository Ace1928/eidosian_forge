import flatbuffers
from flatbuffers.compat import import_numpy
class FloorDivOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        floorDivOptions = FloorDivOptions()
        floorDivOptions.Init(buf, pos)
        return cls.InitFromObj(floorDivOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, floorDivOptions):
        x = FloorDivOptionsT()
        x._UnPack(floorDivOptions)
        return x

    def _UnPack(self, floorDivOptions):
        if floorDivOptions is None:
            return

    def Pack(self, builder):
        FloorDivOptionsStart(builder)
        floorDivOptions = FloorDivOptionsEnd(builder)
        return floorDivOptions