import flatbuffers
from flatbuffers.compat import import_numpy
class VarHandleOptionsT(object):

    def __init__(self):
        self.container = None
        self.sharedName = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        varHandleOptions = VarHandleOptions()
        varHandleOptions.Init(buf, pos)
        return cls.InitFromObj(varHandleOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, varHandleOptions):
        x = VarHandleOptionsT()
        x._UnPack(varHandleOptions)
        return x

    def _UnPack(self, varHandleOptions):
        if varHandleOptions is None:
            return
        self.container = varHandleOptions.Container()
        self.sharedName = varHandleOptions.SharedName()

    def Pack(self, builder):
        if self.container is not None:
            container = builder.CreateString(self.container)
        if self.sharedName is not None:
            sharedName = builder.CreateString(self.sharedName)
        VarHandleOptionsStart(builder)
        if self.container is not None:
            VarHandleOptionsAddContainer(builder, container)
        if self.sharedName is not None:
            VarHandleOptionsAddSharedName(builder, sharedName)
        varHandleOptions = VarHandleOptionsEnd(builder)
        return varHandleOptions