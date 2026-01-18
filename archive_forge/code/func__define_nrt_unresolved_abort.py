from numba.core import config
from numba.core import types, cgutils
from llvmlite import ir, binding
def _define_nrt_unresolved_abort(ctx, module):
    """
    Defines an abort function due to unresolved symbol.

    The function takes no args and will always raise an exception.
    It should be safe to call this function with incorrect number of arguments.
    """
    fnty = ctx.call_conv.get_function_type(types.none, ())
    fn = ir.Function(module, fnty, name='nrt_unresolved_abort')
    bb = fn.append_basic_block()
    builder = ir.IRBuilder(bb)
    msg = 'numba jitted function aborted due to unresolved symbol'
    ctx.call_conv.return_user_exc(builder, RuntimeError, (msg,))
    return fn