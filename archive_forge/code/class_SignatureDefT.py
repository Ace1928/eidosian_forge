import flatbuffers
from flatbuffers.compat import import_numpy
class SignatureDefT(object):

    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.signatureKey = None
        self.subgraphIndex = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        signatureDef = SignatureDef()
        signatureDef.Init(buf, pos)
        return cls.InitFromObj(signatureDef)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, signatureDef):
        x = SignatureDefT()
        x._UnPack(signatureDef)
        return x

    def _UnPack(self, signatureDef):
        if signatureDef is None:
            return
        if not signatureDef.InputsIsNone():
            self.inputs = []
            for i in range(signatureDef.InputsLength()):
                if signatureDef.Inputs(i) is None:
                    self.inputs.append(None)
                else:
                    tensorMap_ = TensorMapT.InitFromObj(signatureDef.Inputs(i))
                    self.inputs.append(tensorMap_)
        if not signatureDef.OutputsIsNone():
            self.outputs = []
            for i in range(signatureDef.OutputsLength()):
                if signatureDef.Outputs(i) is None:
                    self.outputs.append(None)
                else:
                    tensorMap_ = TensorMapT.InitFromObj(signatureDef.Outputs(i))
                    self.outputs.append(tensorMap_)
        self.signatureKey = signatureDef.SignatureKey()
        self.subgraphIndex = signatureDef.SubgraphIndex()

    def Pack(self, builder):
        if self.inputs is not None:
            inputslist = []
            for i in range(len(self.inputs)):
                inputslist.append(self.inputs[i].Pack(builder))
            SignatureDefStartInputsVector(builder, len(self.inputs))
            for i in reversed(range(len(self.inputs))):
                builder.PrependUOffsetTRelative(inputslist[i])
            inputs = builder.EndVector()
        if self.outputs is not None:
            outputslist = []
            for i in range(len(self.outputs)):
                outputslist.append(self.outputs[i].Pack(builder))
            SignatureDefStartOutputsVector(builder, len(self.outputs))
            for i in reversed(range(len(self.outputs))):
                builder.PrependUOffsetTRelative(outputslist[i])
            outputs = builder.EndVector()
        if self.signatureKey is not None:
            signatureKey = builder.CreateString(self.signatureKey)
        SignatureDefStart(builder)
        if self.inputs is not None:
            SignatureDefAddInputs(builder, inputs)
        if self.outputs is not None:
            SignatureDefAddOutputs(builder, outputs)
        if self.signatureKey is not None:
            SignatureDefAddSignatureKey(builder, signatureKey)
        SignatureDefAddSubgraphIndex(builder, self.subgraphIndex)
        signatureDef = SignatureDefEnd(builder)
        return signatureDef