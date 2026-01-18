from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class _LandingPadClause(object):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '{kind} {type} {value}'.format(kind=self.kind, type=self.value.type, value=self.value.get_reference())