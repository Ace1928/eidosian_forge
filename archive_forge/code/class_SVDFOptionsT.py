import flatbuffers
from flatbuffers.compat import import_numpy
class SVDFOptionsT(object):

    def __init__(self):
        self.rank = 0
        self.fusedActivationFunction = 0
        self.asymmetricQuantizeInputs = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        svdfoptions = SVDFOptions()
        svdfoptions.Init(buf, pos)
        return cls.InitFromObj(svdfoptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, svdfoptions):
        x = SVDFOptionsT()
        x._UnPack(svdfoptions)
        return x

    def _UnPack(self, svdfoptions):
        if svdfoptions is None:
            return
        self.rank = svdfoptions.Rank()
        self.fusedActivationFunction = svdfoptions.FusedActivationFunction()
        self.asymmetricQuantizeInputs = svdfoptions.AsymmetricQuantizeInputs()

    def Pack(self, builder):
        SVDFOptionsStart(builder)
        SVDFOptionsAddRank(builder, self.rank)
        SVDFOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        SVDFOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        svdfoptions = SVDFOptionsEnd(builder)
        return svdfoptions