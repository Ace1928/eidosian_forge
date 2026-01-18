from llvmlite import ir
from numba.core import cgutils, types
from numba.core.extending import (intrinsic, make_attribute_wrapper, models,
from numba import float32
def _generate_next_binding(overloadable_function, return_type):
    """
        Generate the overloads for "next_(some type)" functions.
    """

    @intrinsic
    def intrin_NumPyRandomBitGeneratorType_next_ty(tyctx, inst):
        sig = return_type(inst)

        def codegen(cgctx, builder, sig, llargs):
            name = overloadable_function.__name__
            struct_ptr = cgutils.create_struct_proxy(inst)(cgctx, builder, value=llargs[0])
            state = struct_ptr.state
            next_double_addr = getattr(struct_ptr, f'fnptr_{name}')
            ll_void_ptr_t = cgctx.get_value_type(types.voidptr)
            ll_return_t = cgctx.get_value_type(return_type)
            ll_uintp_t = cgctx.get_value_type(types.uintp)
            next_fn_fnptr = builder.inttoptr(next_double_addr, ll_void_ptr_t)
            fnty = ir.FunctionType(ll_return_t, (ll_uintp_t,))
            next_fn = cgutils.get_or_insert_function(builder.module, fnty, name)
            fnptr_as_fntype = builder.bitcast(next_fn_fnptr, next_fn.type)
            ret = builder.call(fnptr_as_fntype, (state,))
            return ret
        return (sig, codegen)

    @overload(overloadable_function)
    def ol_next_ty(bitgen):
        if isinstance(bitgen, types.NumPyRandomBitGeneratorType):

            def impl(bitgen):
                return intrin_NumPyRandomBitGeneratorType_next_ty(bitgen)
            return impl