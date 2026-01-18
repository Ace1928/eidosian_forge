import flatbuffers
from flatbuffers.compat import import_numpy
class DepthToSpaceOptionsT(object):

    def __init__(self):
        self.blockSize = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        depthToSpaceOptions = DepthToSpaceOptions()
        depthToSpaceOptions.Init(buf, pos)
        return cls.InitFromObj(depthToSpaceOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, depthToSpaceOptions):
        x = DepthToSpaceOptionsT()
        x._UnPack(depthToSpaceOptions)
        return x

    def _UnPack(self, depthToSpaceOptions):
        if depthToSpaceOptions is None:
            return
        self.blockSize = depthToSpaceOptions.BlockSize()

    def Pack(self, builder):
        DepthToSpaceOptionsStart(builder)
        DepthToSpaceOptionsAddBlockSize(builder, self.blockSize)
        depthToSpaceOptions = DepthToSpaceOptionsEnd(builder)
        return depthToSpaceOptions