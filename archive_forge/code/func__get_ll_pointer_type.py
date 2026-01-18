import struct
from llvmlite.ir._utils import _StrCaching
def _get_ll_pointer_type(self, target_data, context=None):
    """
        Convert this type object to an LLVM type.
        """
    from llvmlite.ir import Module, GlobalVariable
    from llvmlite.binding import parse_assembly
    if context is None:
        m = Module()
    else:
        m = Module(context=context)
    foo = GlobalVariable(m, self, name='foo')
    with parse_assembly(str(m)) as llmod:
        return llmod.get_global_variable(foo.name).type