import flatbuffers
from flatbuffers.compat import import_numpy
class SqueezeOptionsT(object):

    def __init__(self):
        self.squeezeDims = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        squeezeOptions = SqueezeOptions()
        squeezeOptions.Init(buf, pos)
        return cls.InitFromObj(squeezeOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, squeezeOptions):
        x = SqueezeOptionsT()
        x._UnPack(squeezeOptions)
        return x

    def _UnPack(self, squeezeOptions):
        if squeezeOptions is None:
            return
        if not squeezeOptions.SqueezeDimsIsNone():
            if np is None:
                self.squeezeDims = []
                for i in range(squeezeOptions.SqueezeDimsLength()):
                    self.squeezeDims.append(squeezeOptions.SqueezeDims(i))
            else:
                self.squeezeDims = squeezeOptions.SqueezeDimsAsNumpy()

    def Pack(self, builder):
        if self.squeezeDims is not None:
            if np is not None and type(self.squeezeDims) is np.ndarray:
                squeezeDims = builder.CreateNumpyVector(self.squeezeDims)
            else:
                SqueezeOptionsStartSqueezeDimsVector(builder, len(self.squeezeDims))
                for i in reversed(range(len(self.squeezeDims))):
                    builder.PrependInt32(self.squeezeDims[i])
                squeezeDims = builder.EndVector()
        SqueezeOptionsStart(builder)
        if self.squeezeDims is not None:
            SqueezeOptionsAddSqueezeDims(builder, squeezeDims)
        squeezeOptions = SqueezeOptionsEnd(builder)
        return squeezeOptions