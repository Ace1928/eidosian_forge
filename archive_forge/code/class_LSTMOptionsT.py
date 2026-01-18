import flatbuffers
from flatbuffers.compat import import_numpy
class LSTMOptionsT(object):

    def __init__(self):
        self.fusedActivationFunction = 0
        self.cellClip = 0.0
        self.projClip = 0.0
        self.kernelType = 0
        self.asymmetricQuantizeInputs = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        lstmoptions = LSTMOptions()
        lstmoptions.Init(buf, pos)
        return cls.InitFromObj(lstmoptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, lstmoptions):
        x = LSTMOptionsT()
        x._UnPack(lstmoptions)
        return x

    def _UnPack(self, lstmoptions):
        if lstmoptions is None:
            return
        self.fusedActivationFunction = lstmoptions.FusedActivationFunction()
        self.cellClip = lstmoptions.CellClip()
        self.projClip = lstmoptions.ProjClip()
        self.kernelType = lstmoptions.KernelType()
        self.asymmetricQuantizeInputs = lstmoptions.AsymmetricQuantizeInputs()

    def Pack(self, builder):
        LSTMOptionsStart(builder)
        LSTMOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        LSTMOptionsAddCellClip(builder, self.cellClip)
        LSTMOptionsAddProjClip(builder, self.projClip)
        LSTMOptionsAddKernelType(builder, self.kernelType)
        LSTMOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        lstmoptions = LSTMOptionsEnd(builder)
        return lstmoptions