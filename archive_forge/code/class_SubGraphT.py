import flatbuffers
from flatbuffers.compat import import_numpy
class SubGraphT(object):

    def __init__(self):
        self.tensors = None
        self.inputs = None
        self.outputs = None
        self.operators = None
        self.name = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        subGraph = SubGraph()
        subGraph.Init(buf, pos)
        return cls.InitFromObj(subGraph)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, subGraph):
        x = SubGraphT()
        x._UnPack(subGraph)
        return x

    def _UnPack(self, subGraph):
        if subGraph is None:
            return
        if not subGraph.TensorsIsNone():
            self.tensors = []
            for i in range(subGraph.TensorsLength()):
                if subGraph.Tensors(i) is None:
                    self.tensors.append(None)
                else:
                    tensor_ = TensorT.InitFromObj(subGraph.Tensors(i))
                    self.tensors.append(tensor_)
        if not subGraph.InputsIsNone():
            if np is None:
                self.inputs = []
                for i in range(subGraph.InputsLength()):
                    self.inputs.append(subGraph.Inputs(i))
            else:
                self.inputs = subGraph.InputsAsNumpy()
        if not subGraph.OutputsIsNone():
            if np is None:
                self.outputs = []
                for i in range(subGraph.OutputsLength()):
                    self.outputs.append(subGraph.Outputs(i))
            else:
                self.outputs = subGraph.OutputsAsNumpy()
        if not subGraph.OperatorsIsNone():
            self.operators = []
            for i in range(subGraph.OperatorsLength()):
                if subGraph.Operators(i) is None:
                    self.operators.append(None)
                else:
                    operator_ = OperatorT.InitFromObj(subGraph.Operators(i))
                    self.operators.append(operator_)
        self.name = subGraph.Name()

    def Pack(self, builder):
        if self.tensors is not None:
            tensorslist = []
            for i in range(len(self.tensors)):
                tensorslist.append(self.tensors[i].Pack(builder))
            SubGraphStartTensorsVector(builder, len(self.tensors))
            for i in reversed(range(len(self.tensors))):
                builder.PrependUOffsetTRelative(tensorslist[i])
            tensors = builder.EndVector()
        if self.inputs is not None:
            if np is not None and type(self.inputs) is np.ndarray:
                inputs = builder.CreateNumpyVector(self.inputs)
            else:
                SubGraphStartInputsVector(builder, len(self.inputs))
                for i in reversed(range(len(self.inputs))):
                    builder.PrependInt32(self.inputs[i])
                inputs = builder.EndVector()
        if self.outputs is not None:
            if np is not None and type(self.outputs) is np.ndarray:
                outputs = builder.CreateNumpyVector(self.outputs)
            else:
                SubGraphStartOutputsVector(builder, len(self.outputs))
                for i in reversed(range(len(self.outputs))):
                    builder.PrependInt32(self.outputs[i])
                outputs = builder.EndVector()
        if self.operators is not None:
            operatorslist = []
            for i in range(len(self.operators)):
                operatorslist.append(self.operators[i].Pack(builder))
            SubGraphStartOperatorsVector(builder, len(self.operators))
            for i in reversed(range(len(self.operators))):
                builder.PrependUOffsetTRelative(operatorslist[i])
            operators = builder.EndVector()
        if self.name is not None:
            name = builder.CreateString(self.name)
        SubGraphStart(builder)
        if self.tensors is not None:
            SubGraphAddTensors(builder, tensors)
        if self.inputs is not None:
            SubGraphAddInputs(builder, inputs)
        if self.outputs is not None:
            SubGraphAddOutputs(builder, outputs)
        if self.operators is not None:
            SubGraphAddOperators(builder, operators)
        if self.name is not None:
            SubGraphAddName(builder, name)
        subGraph = SubGraphEnd(builder)
        return subGraph