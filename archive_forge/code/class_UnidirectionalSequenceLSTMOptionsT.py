import flatbuffers
from flatbuffers.compat import import_numpy
class UnidirectionalSequenceLSTMOptionsT(object):

    def __init__(self):
        self.fusedActivationFunction = 0
        self.cellClip = 0.0
        self.projClip = 0.0
        self.timeMajor = False
        self.asymmetricQuantizeInputs = False
        self.diagonalRecurrentTensors = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        unidirectionalSequenceLstmoptions = UnidirectionalSequenceLSTMOptions()
        unidirectionalSequenceLstmoptions.Init(buf, pos)
        return cls.InitFromObj(unidirectionalSequenceLstmoptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, unidirectionalSequenceLstmoptions):
        x = UnidirectionalSequenceLSTMOptionsT()
        x._UnPack(unidirectionalSequenceLstmoptions)
        return x

    def _UnPack(self, unidirectionalSequenceLstmoptions):
        if unidirectionalSequenceLstmoptions is None:
            return
        self.fusedActivationFunction = unidirectionalSequenceLstmoptions.FusedActivationFunction()
        self.cellClip = unidirectionalSequenceLstmoptions.CellClip()
        self.projClip = unidirectionalSequenceLstmoptions.ProjClip()
        self.timeMajor = unidirectionalSequenceLstmoptions.TimeMajor()
        self.asymmetricQuantizeInputs = unidirectionalSequenceLstmoptions.AsymmetricQuantizeInputs()
        self.diagonalRecurrentTensors = unidirectionalSequenceLstmoptions.DiagonalRecurrentTensors()

    def Pack(self, builder):
        UnidirectionalSequenceLSTMOptionsStart(builder)
        UnidirectionalSequenceLSTMOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        UnidirectionalSequenceLSTMOptionsAddCellClip(builder, self.cellClip)
        UnidirectionalSequenceLSTMOptionsAddProjClip(builder, self.projClip)
        UnidirectionalSequenceLSTMOptionsAddTimeMajor(builder, self.timeMajor)
        UnidirectionalSequenceLSTMOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        UnidirectionalSequenceLSTMOptionsAddDiagonalRecurrentTensors(builder, self.diagonalRecurrentTensors)
        unidirectionalSequenceLstmoptions = UnidirectionalSequenceLSTMOptionsEnd(builder)
        return unidirectionalSequenceLstmoptions