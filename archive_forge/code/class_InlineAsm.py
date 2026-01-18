from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class InlineAsm(object):

    def __init__(self, ftype, asm, constraint, side_effect=False):
        self.type = ftype.return_type
        self.function_type = ftype
        self.asm = asm
        self.constraint = constraint
        self.side_effect = side_effect

    def descr(self, buf):
        sideeffect = 'sideeffect' if self.side_effect else ''
        fmt = 'asm {sideeffect} "{asm}", "{constraint}"\n'
        buf.append(fmt.format(sideeffect=sideeffect, asm=self.asm, constraint=self.constraint))

    def get_reference(self):
        buf = []
        self.descr(buf)
        return ''.join(buf)

    def __str__(self):
        return '{0} {1}'.format(self.type, self.get_reference())