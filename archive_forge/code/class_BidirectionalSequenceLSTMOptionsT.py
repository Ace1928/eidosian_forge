import flatbuffers
from flatbuffers.compat import import_numpy
class BidirectionalSequenceLSTMOptionsT(object):

    def __init__(self):
        self.fusedActivationFunction = 0
        self.cellClip = 0.0
        self.projClip = 0.0
        self.mergeOutputs = False
        self.timeMajor = True
        self.asymmetricQuantizeInputs = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        bidirectionalSequenceLstmoptions = BidirectionalSequenceLSTMOptions()
        bidirectionalSequenceLstmoptions.Init(buf, pos)
        return cls.InitFromObj(bidirectionalSequenceLstmoptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, bidirectionalSequenceLstmoptions):
        x = BidirectionalSequenceLSTMOptionsT()
        x._UnPack(bidirectionalSequenceLstmoptions)
        return x

    def _UnPack(self, bidirectionalSequenceLstmoptions):
        if bidirectionalSequenceLstmoptions is None:
            return
        self.fusedActivationFunction = bidirectionalSequenceLstmoptions.FusedActivationFunction()
        self.cellClip = bidirectionalSequenceLstmoptions.CellClip()
        self.projClip = bidirectionalSequenceLstmoptions.ProjClip()
        self.mergeOutputs = bidirectionalSequenceLstmoptions.MergeOutputs()
        self.timeMajor = bidirectionalSequenceLstmoptions.TimeMajor()
        self.asymmetricQuantizeInputs = bidirectionalSequenceLstmoptions.AsymmetricQuantizeInputs()

    def Pack(self, builder):
        BidirectionalSequenceLSTMOptionsStart(builder)
        BidirectionalSequenceLSTMOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        BidirectionalSequenceLSTMOptionsAddCellClip(builder, self.cellClip)
        BidirectionalSequenceLSTMOptionsAddProjClip(builder, self.projClip)
        BidirectionalSequenceLSTMOptionsAddMergeOutputs(builder, self.mergeOutputs)
        BidirectionalSequenceLSTMOptionsAddTimeMajor(builder, self.timeMajor)
        BidirectionalSequenceLSTMOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        bidirectionalSequenceLstmoptions = BidirectionalSequenceLSTMOptionsEnd(builder)
        return bidirectionalSequenceLstmoptions