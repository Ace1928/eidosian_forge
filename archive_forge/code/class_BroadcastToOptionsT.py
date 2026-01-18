import flatbuffers
from flatbuffers.compat import import_numpy
class BroadcastToOptionsT(object):

    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        broadcastToOptions = BroadcastToOptions()
        broadcastToOptions.Init(buf, pos)
        return cls.InitFromObj(broadcastToOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, broadcastToOptions):
        x = BroadcastToOptionsT()
        x._UnPack(broadcastToOptions)
        return x

    def _UnPack(self, broadcastToOptions):
        if broadcastToOptions is None:
            return

    def Pack(self, builder):
        BroadcastToOptionsStart(builder)
        broadcastToOptions = BroadcastToOptionsEnd(builder)
        return broadcastToOptions