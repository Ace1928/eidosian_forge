import flatbuffers
from flatbuffers.compat import import_numpy
class SpaceToDepthOptionsT(object):

    def __init__(self):
        self.blockSize = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        spaceToDepthOptions = SpaceToDepthOptions()
        spaceToDepthOptions.Init(buf, pos)
        return cls.InitFromObj(spaceToDepthOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, spaceToDepthOptions):
        x = SpaceToDepthOptionsT()
        x._UnPack(spaceToDepthOptions)
        return x

    def _UnPack(self, spaceToDepthOptions):
        if spaceToDepthOptions is None:
            return
        self.blockSize = spaceToDepthOptions.BlockSize()

    def Pack(self, builder):
        SpaceToDepthOptionsStart(builder)
        SpaceToDepthOptionsAddBlockSize(builder, self.blockSize)
        spaceToDepthOptions = SpaceToDepthOptionsEnd(builder)
        return spaceToDepthOptions