import flatbuffers
from flatbuffers.compat import import_numpy
class OperatorCodeT(object):

    def __init__(self):
        self.deprecatedBuiltinCode = 0
        self.customCode = None
        self.version = 1
        self.builtinCode = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        operatorCode = OperatorCode()
        operatorCode.Init(buf, pos)
        return cls.InitFromObj(operatorCode)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, operatorCode):
        x = OperatorCodeT()
        x._UnPack(operatorCode)
        return x

    def _UnPack(self, operatorCode):
        if operatorCode is None:
            return
        self.deprecatedBuiltinCode = operatorCode.DeprecatedBuiltinCode()
        self.customCode = operatorCode.CustomCode()
        self.version = operatorCode.Version()
        self.builtinCode = operatorCode.BuiltinCode()

    def Pack(self, builder):
        if self.customCode is not None:
            customCode = builder.CreateString(self.customCode)
        OperatorCodeStart(builder)
        OperatorCodeAddDeprecatedBuiltinCode(builder, self.deprecatedBuiltinCode)
        if self.customCode is not None:
            OperatorCodeAddCustomCode(builder, customCode)
        OperatorCodeAddVersion(builder, self.version)
        OperatorCodeAddBuiltinCode(builder, self.builtinCode)
        operatorCode = OperatorCodeEnd(builder)
        return operatorCode