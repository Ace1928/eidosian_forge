import flatbuffers
from flatbuffers.compat import import_numpy
class OperatorT(object):

    def __init__(self):
        self.opcodeIndex = 0
        self.inputs = None
        self.outputs = None
        self.builtinOptionsType = 0
        self.builtinOptions = None
        self.customOptions = None
        self.customOptionsFormat = 0
        self.mutatingVariableInputs = None
        self.intermediates = None
        self.largeCustomOptionsOffset = 0
        self.largeCustomOptionsSize = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        operator = Operator()
        operator.Init(buf, pos)
        return cls.InitFromObj(operator)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, operator):
        x = OperatorT()
        x._UnPack(operator)
        return x

    def _UnPack(self, operator):
        if operator is None:
            return
        self.opcodeIndex = operator.OpcodeIndex()
        if not operator.InputsIsNone():
            if np is None:
                self.inputs = []
                for i in range(operator.InputsLength()):
                    self.inputs.append(operator.Inputs(i))
            else:
                self.inputs = operator.InputsAsNumpy()
        if not operator.OutputsIsNone():
            if np is None:
                self.outputs = []
                for i in range(operator.OutputsLength()):
                    self.outputs.append(operator.Outputs(i))
            else:
                self.outputs = operator.OutputsAsNumpy()
        self.builtinOptionsType = operator.BuiltinOptionsType()
        self.builtinOptions = BuiltinOptionsCreator(self.builtinOptionsType, operator.BuiltinOptions())
        if not operator.CustomOptionsIsNone():
            if np is None:
                self.customOptions = []
                for i in range(operator.CustomOptionsLength()):
                    self.customOptions.append(operator.CustomOptions(i))
            else:
                self.customOptions = operator.CustomOptionsAsNumpy()
        self.customOptionsFormat = operator.CustomOptionsFormat()
        if not operator.MutatingVariableInputsIsNone():
            if np is None:
                self.mutatingVariableInputs = []
                for i in range(operator.MutatingVariableInputsLength()):
                    self.mutatingVariableInputs.append(operator.MutatingVariableInputs(i))
            else:
                self.mutatingVariableInputs = operator.MutatingVariableInputsAsNumpy()
        if not operator.IntermediatesIsNone():
            if np is None:
                self.intermediates = []
                for i in range(operator.IntermediatesLength()):
                    self.intermediates.append(operator.Intermediates(i))
            else:
                self.intermediates = operator.IntermediatesAsNumpy()
        self.largeCustomOptionsOffset = operator.LargeCustomOptionsOffset()
        self.largeCustomOptionsSize = operator.LargeCustomOptionsSize()

    def Pack(self, builder):
        if self.inputs is not None:
            if np is not None and type(self.inputs) is np.ndarray:
                inputs = builder.CreateNumpyVector(self.inputs)
            else:
                OperatorStartInputsVector(builder, len(self.inputs))
                for i in reversed(range(len(self.inputs))):
                    builder.PrependInt32(self.inputs[i])
                inputs = builder.EndVector()
        if self.outputs is not None:
            if np is not None and type(self.outputs) is np.ndarray:
                outputs = builder.CreateNumpyVector(self.outputs)
            else:
                OperatorStartOutputsVector(builder, len(self.outputs))
                for i in reversed(range(len(self.outputs))):
                    builder.PrependInt32(self.outputs[i])
                outputs = builder.EndVector()
        if self.builtinOptions is not None:
            builtinOptions = self.builtinOptions.Pack(builder)
        if self.customOptions is not None:
            if np is not None and type(self.customOptions) is np.ndarray:
                customOptions = builder.CreateNumpyVector(self.customOptions)
            else:
                OperatorStartCustomOptionsVector(builder, len(self.customOptions))
                for i in reversed(range(len(self.customOptions))):
                    builder.PrependUint8(self.customOptions[i])
                customOptions = builder.EndVector()
        if self.mutatingVariableInputs is not None:
            if np is not None and type(self.mutatingVariableInputs) is np.ndarray:
                mutatingVariableInputs = builder.CreateNumpyVector(self.mutatingVariableInputs)
            else:
                OperatorStartMutatingVariableInputsVector(builder, len(self.mutatingVariableInputs))
                for i in reversed(range(len(self.mutatingVariableInputs))):
                    builder.PrependBool(self.mutatingVariableInputs[i])
                mutatingVariableInputs = builder.EndVector()
        if self.intermediates is not None:
            if np is not None and type(self.intermediates) is np.ndarray:
                intermediates = builder.CreateNumpyVector(self.intermediates)
            else:
                OperatorStartIntermediatesVector(builder, len(self.intermediates))
                for i in reversed(range(len(self.intermediates))):
                    builder.PrependInt32(self.intermediates[i])
                intermediates = builder.EndVector()
        OperatorStart(builder)
        OperatorAddOpcodeIndex(builder, self.opcodeIndex)
        if self.inputs is not None:
            OperatorAddInputs(builder, inputs)
        if self.outputs is not None:
            OperatorAddOutputs(builder, outputs)
        OperatorAddBuiltinOptionsType(builder, self.builtinOptionsType)
        if self.builtinOptions is not None:
            OperatorAddBuiltinOptions(builder, builtinOptions)
        if self.customOptions is not None:
            OperatorAddCustomOptions(builder, customOptions)
        OperatorAddCustomOptionsFormat(builder, self.customOptionsFormat)
        if self.mutatingVariableInputs is not None:
            OperatorAddMutatingVariableInputs(builder, mutatingVariableInputs)
        if self.intermediates is not None:
            OperatorAddIntermediates(builder, intermediates)
        OperatorAddLargeCustomOptionsOffset(builder, self.largeCustomOptionsOffset)
        OperatorAddLargeCustomOptionsSize(builder, self.largeCustomOptionsSize)
        operator = OperatorEnd(builder)
        return operator