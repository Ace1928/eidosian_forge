import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorCodeAddBuiltinCode(builder, builtinCode):
    builder.PrependInt32Slot(3, builtinCode, 0)