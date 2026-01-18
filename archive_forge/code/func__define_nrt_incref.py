from numba.core import config
from numba.core import types, cgutils
from llvmlite import ir, binding
def _define_nrt_incref(module, atomic_incr):
    """
    Implement NRT_incref in the module
    """
    fn_incref = cgutils.get_or_insert_function(module, incref_decref_ty, 'NRT_incref')
    fn_incref.attributes.add('noinline')
    builder = ir.IRBuilder(fn_incref.append_basic_block())
    [ptr] = fn_incref.args
    is_null = builder.icmp_unsigned('==', ptr, cgutils.get_null_value(ptr.type))
    with cgutils.if_unlikely(builder, is_null):
        builder.ret_void()
    word_ptr = builder.bitcast(ptr, atomic_incr.args[0].type)
    if config.DEBUG_NRT:
        cgutils.printf(builder, '*** NRT_Incref %zu [%p]\n', builder.load(word_ptr), ptr)
    builder.call(atomic_incr, [word_ptr])
    builder.ret_void()