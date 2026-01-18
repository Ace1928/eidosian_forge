import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class BlockAddress(Value):
    """
    The address of a basic block.
    """

    def __init__(self, function, basic_block):
        assert isinstance(function, Function)
        assert isinstance(basic_block, Block)
        self.type = types.IntType(8).as_pointer()
        self.function = function
        self.basic_block = basic_block

    def __str__(self):
        return '{0} {1}'.format(self.type, self.get_reference())

    def get_reference(self):
        return 'blockaddress({0}, {1})'.format(self.function.get_reference(), self.basic_block.get_reference())