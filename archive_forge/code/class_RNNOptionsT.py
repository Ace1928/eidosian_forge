import flatbuffers
from flatbuffers.compat import import_numpy
class RNNOptionsT(object):

    def __init__(self):
        self.fusedActivationFunction = 0
        self.asymmetricQuantizeInputs = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        rnnoptions = RNNOptions()
        rnnoptions.Init(buf, pos)
        return cls.InitFromObj(rnnoptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, rnnoptions):
        x = RNNOptionsT()
        x._UnPack(rnnoptions)
        return x

    def _UnPack(self, rnnoptions):
        if rnnoptions is None:
            return
        self.fusedActivationFunction = rnnoptions.FusedActivationFunction()
        self.asymmetricQuantizeInputs = rnnoptions.AsymmetricQuantizeInputs()

    def Pack(self, builder):
        RNNOptionsStart(builder)
        RNNOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        RNNOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        rnnoptions = RNNOptionsEnd(builder)
        return rnnoptions