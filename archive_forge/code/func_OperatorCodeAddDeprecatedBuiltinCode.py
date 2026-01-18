import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorCodeAddDeprecatedBuiltinCode(builder, deprecatedBuiltinCode):
    builder.PrependInt8Slot(0, deprecatedBuiltinCode, 0)