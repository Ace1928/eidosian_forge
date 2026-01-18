import operator
import warnings
from llvmlite import ir
from numba.core import types, cgutils
from numba.core import typing
from numba.core.registry import cpu_target
from numba.core.typeconv import Conversion
from numba.core.extending import intrinsic
from numba.core.errors import TypingError, NumbaTypeSafetyWarning
def _get_equal(context, module, datamodel, container_element_type):
    assert datamodel.contains_nrt_meminfo()
    fe_type = datamodel.fe_type
    data_ptr_ty = datamodel.get_data_type().as_pointer()
    wrapfnty = context.call_conv.get_function_type(types.int32, [fe_type, fe_type])
    argtypes = [fe_type, fe_type]

    def build_wrapper(fn):
        builder = ir.IRBuilder(fn.append_basic_block())
        args = context.call_conv.decode_arguments(builder, argtypes, fn)
        sig = typing.signature(types.boolean, fe_type, fe_type)
        op = operator.eq
        fnop = context.typing_context.resolve_value_type(op)
        fnop.get_call_type(context.typing_context, sig.args, {})
        eqfn = context.get_function(fnop, sig)
        res = eqfn(builder, args)
        intres = context.cast(builder, res, types.boolean, types.int32)
        context.call_conv.return_value(builder, intres)
    wrapfn = cgutils.get_or_insert_function(module, wrapfnty, name='.numba_{}.{}_equal.wrap'.format(context.fndesc.mangled_name, container_element_type))
    build_wrapper(wrapfn)
    equal_fnty = ir.FunctionType(ir.IntType(32), [data_ptr_ty, data_ptr_ty])
    equal_fn = cgutils.get_or_insert_function(module, equal_fnty, name='.numba_{}.{}_equal'.format(context.fndesc.mangled_name, container_element_type))
    builder = ir.IRBuilder(equal_fn.append_basic_block())
    lhs = datamodel.load_from_data_pointer(builder, equal_fn.args[0])
    rhs = datamodel.load_from_data_pointer(builder, equal_fn.args[1])
    status, retval = context.call_conv.call_function(builder, wrapfn, types.boolean, argtypes, [lhs, rhs])
    with builder.if_then(status.is_ok, likely=True):
        with builder.if_then(status.is_none):
            builder.ret(context.get_constant(types.int32, 0))
        retval = context.cast(builder, retval, types.boolean, types.int32)
        builder.ret(retval)
    builder.ret(context.get_constant(types.int32, -1))
    return equal_fn