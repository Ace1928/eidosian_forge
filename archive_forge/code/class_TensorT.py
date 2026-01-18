import flatbuffers
from flatbuffers.compat import import_numpy
class TensorT(object):

    def __init__(self):
        self.shape = None
        self.type = 0
        self.buffer = 0
        self.name = None
        self.quantization = None
        self.isVariable = False
        self.sparsity = None
        self.shapeSignature = None
        self.hasRank = False
        self.variantTensors = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        tensor = Tensor()
        tensor.Init(buf, pos)
        return cls.InitFromObj(tensor)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, tensor):
        x = TensorT()
        x._UnPack(tensor)
        return x

    def _UnPack(self, tensor):
        if tensor is None:
            return
        if not tensor.ShapeIsNone():
            if np is None:
                self.shape = []
                for i in range(tensor.ShapeLength()):
                    self.shape.append(tensor.Shape(i))
            else:
                self.shape = tensor.ShapeAsNumpy()
        self.type = tensor.Type()
        self.buffer = tensor.Buffer()
        self.name = tensor.Name()
        if tensor.Quantization() is not None:
            self.quantization = QuantizationParametersT.InitFromObj(tensor.Quantization())
        self.isVariable = tensor.IsVariable()
        if tensor.Sparsity() is not None:
            self.sparsity = SparsityParametersT.InitFromObj(tensor.Sparsity())
        if not tensor.ShapeSignatureIsNone():
            if np is None:
                self.shapeSignature = []
                for i in range(tensor.ShapeSignatureLength()):
                    self.shapeSignature.append(tensor.ShapeSignature(i))
            else:
                self.shapeSignature = tensor.ShapeSignatureAsNumpy()
        self.hasRank = tensor.HasRank()
        if not tensor.VariantTensorsIsNone():
            self.variantTensors = []
            for i in range(tensor.VariantTensorsLength()):
                if tensor.VariantTensors(i) is None:
                    self.variantTensors.append(None)
                else:
                    variantSubType_ = VariantSubTypeT.InitFromObj(tensor.VariantTensors(i))
                    self.variantTensors.append(variantSubType_)

    def Pack(self, builder):
        if self.shape is not None:
            if np is not None and type(self.shape) is np.ndarray:
                shape = builder.CreateNumpyVector(self.shape)
            else:
                TensorStartShapeVector(builder, len(self.shape))
                for i in reversed(range(len(self.shape))):
                    builder.PrependInt32(self.shape[i])
                shape = builder.EndVector()
        if self.name is not None:
            name = builder.CreateString(self.name)
        if self.quantization is not None:
            quantization = self.quantization.Pack(builder)
        if self.sparsity is not None:
            sparsity = self.sparsity.Pack(builder)
        if self.shapeSignature is not None:
            if np is not None and type(self.shapeSignature) is np.ndarray:
                shapeSignature = builder.CreateNumpyVector(self.shapeSignature)
            else:
                TensorStartShapeSignatureVector(builder, len(self.shapeSignature))
                for i in reversed(range(len(self.shapeSignature))):
                    builder.PrependInt32(self.shapeSignature[i])
                shapeSignature = builder.EndVector()
        if self.variantTensors is not None:
            variantTensorslist = []
            for i in range(len(self.variantTensors)):
                variantTensorslist.append(self.variantTensors[i].Pack(builder))
            TensorStartVariantTensorsVector(builder, len(self.variantTensors))
            for i in reversed(range(len(self.variantTensors))):
                builder.PrependUOffsetTRelative(variantTensorslist[i])
            variantTensors = builder.EndVector()
        TensorStart(builder)
        if self.shape is not None:
            TensorAddShape(builder, shape)
        TensorAddType(builder, self.type)
        TensorAddBuffer(builder, self.buffer)
        if self.name is not None:
            TensorAddName(builder, name)
        if self.quantization is not None:
            TensorAddQuantization(builder, quantization)
        TensorAddIsVariable(builder, self.isVariable)
        if self.sparsity is not None:
            TensorAddSparsity(builder, sparsity)
        if self.shapeSignature is not None:
            TensorAddShapeSignature(builder, shapeSignature)
        TensorAddHasRank(builder, self.hasRank)
        if self.variantTensors is not None:
            TensorAddVariantTensors(builder, variantTensors)
        tensor = TensorEnd(builder)
        return tensor