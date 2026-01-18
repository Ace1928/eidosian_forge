import flatbuffers
from flatbuffers.compat import import_numpy
class QuantizationParametersT(object):

    def __init__(self):
        self.min = None
        self.max = None
        self.scale = None
        self.zeroPoint = None
        self.detailsType = 0
        self.details = None
        self.quantizedDimension = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        quantizationParameters = QuantizationParameters()
        quantizationParameters.Init(buf, pos)
        return cls.InitFromObj(quantizationParameters)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, quantizationParameters):
        x = QuantizationParametersT()
        x._UnPack(quantizationParameters)
        return x

    def _UnPack(self, quantizationParameters):
        if quantizationParameters is None:
            return
        if not quantizationParameters.MinIsNone():
            if np is None:
                self.min = []
                for i in range(quantizationParameters.MinLength()):
                    self.min.append(quantizationParameters.Min(i))
            else:
                self.min = quantizationParameters.MinAsNumpy()
        if not quantizationParameters.MaxIsNone():
            if np is None:
                self.max = []
                for i in range(quantizationParameters.MaxLength()):
                    self.max.append(quantizationParameters.Max(i))
            else:
                self.max = quantizationParameters.MaxAsNumpy()
        if not quantizationParameters.ScaleIsNone():
            if np is None:
                self.scale = []
                for i in range(quantizationParameters.ScaleLength()):
                    self.scale.append(quantizationParameters.Scale(i))
            else:
                self.scale = quantizationParameters.ScaleAsNumpy()
        if not quantizationParameters.ZeroPointIsNone():
            if np is None:
                self.zeroPoint = []
                for i in range(quantizationParameters.ZeroPointLength()):
                    self.zeroPoint.append(quantizationParameters.ZeroPoint(i))
            else:
                self.zeroPoint = quantizationParameters.ZeroPointAsNumpy()
        self.detailsType = quantizationParameters.DetailsType()
        self.details = QuantizationDetailsCreator(self.detailsType, quantizationParameters.Details())
        self.quantizedDimension = quantizationParameters.QuantizedDimension()

    def Pack(self, builder):
        if self.min is not None:
            if np is not None and type(self.min) is np.ndarray:
                min = builder.CreateNumpyVector(self.min)
            else:
                QuantizationParametersStartMinVector(builder, len(self.min))
                for i in reversed(range(len(self.min))):
                    builder.PrependFloat32(self.min[i])
                min = builder.EndVector()
        if self.max is not None:
            if np is not None and type(self.max) is np.ndarray:
                max = builder.CreateNumpyVector(self.max)
            else:
                QuantizationParametersStartMaxVector(builder, len(self.max))
                for i in reversed(range(len(self.max))):
                    builder.PrependFloat32(self.max[i])
                max = builder.EndVector()
        if self.scale is not None:
            if np is not None and type(self.scale) is np.ndarray:
                scale = builder.CreateNumpyVector(self.scale)
            else:
                QuantizationParametersStartScaleVector(builder, len(self.scale))
                for i in reversed(range(len(self.scale))):
                    builder.PrependFloat32(self.scale[i])
                scale = builder.EndVector()
        if self.zeroPoint is not None:
            if np is not None and type(self.zeroPoint) is np.ndarray:
                zeroPoint = builder.CreateNumpyVector(self.zeroPoint)
            else:
                QuantizationParametersStartZeroPointVector(builder, len(self.zeroPoint))
                for i in reversed(range(len(self.zeroPoint))):
                    builder.PrependInt64(self.zeroPoint[i])
                zeroPoint = builder.EndVector()
        if self.details is not None:
            details = self.details.Pack(builder)
        QuantizationParametersStart(builder)
        if self.min is not None:
            QuantizationParametersAddMin(builder, min)
        if self.max is not None:
            QuantizationParametersAddMax(builder, max)
        if self.scale is not None:
            QuantizationParametersAddScale(builder, scale)
        if self.zeroPoint is not None:
            QuantizationParametersAddZeroPoint(builder, zeroPoint)
        QuantizationParametersAddDetailsType(builder, self.detailsType)
        if self.details is not None:
            QuantizationParametersAddDetails(builder, details)
        QuantizationParametersAddQuantizedDimension(builder, self.quantizedDimension)
        quantizationParameters = QuantizationParametersEnd(builder)
        return quantizationParameters