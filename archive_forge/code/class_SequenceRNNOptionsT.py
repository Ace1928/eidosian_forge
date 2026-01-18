import flatbuffers
from flatbuffers.compat import import_numpy
class SequenceRNNOptionsT(object):

    def __init__(self):
        self.timeMajor = False
        self.fusedActivationFunction = 0
        self.asymmetricQuantizeInputs = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sequenceRnnoptions = SequenceRNNOptions()
        sequenceRnnoptions.Init(buf, pos)
        return cls.InitFromObj(sequenceRnnoptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, sequenceRnnoptions):
        x = SequenceRNNOptionsT()
        x._UnPack(sequenceRnnoptions)
        return x

    def _UnPack(self, sequenceRnnoptions):
        if sequenceRnnoptions is None:
            return
        self.timeMajor = sequenceRnnoptions.TimeMajor()
        self.fusedActivationFunction = sequenceRnnoptions.FusedActivationFunction()
        self.asymmetricQuantizeInputs = sequenceRnnoptions.AsymmetricQuantizeInputs()

    def Pack(self, builder):
        SequenceRNNOptionsStart(builder)
        SequenceRNNOptionsAddTimeMajor(builder, self.timeMajor)
        SequenceRNNOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        SequenceRNNOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        sequenceRnnoptions = SequenceRNNOptionsEnd(builder)
        return sequenceRnnoptions