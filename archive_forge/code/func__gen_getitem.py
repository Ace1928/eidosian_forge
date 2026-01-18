import operator
from enum import IntEnum
from llvmlite import ir
from numba.core.extending import (
from numba.core.imputils import iternext_impl
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
from numba.cpython import listobj
def _gen_getitem(borrowed):

    @intrinsic
    def impl(typingctx, l_ty, index_ty):
        is_none = isinstance(l_ty.item_type, types.NoneType)
        if is_none:
            resty = types.Tuple([types.int32, l_ty.item_type])
        else:
            resty = types.Tuple([types.int32, types.Optional(l_ty.item_type)])
        sig = resty(l_ty, index_ty)

        def codegen(context, builder, sig, args):
            [tl, tindex] = sig.args
            [l, index] = args
            fnty = ir.FunctionType(ll_voidptr_type, [ll_list_type])
            fname = 'numba_list_base_ptr'
            fn = cgutils.get_or_insert_function(builder.module, fnty, fname)
            fn.attributes.add('alwaysinline')
            fn.attributes.add('nounwind')
            fn.attributes.add('readonly')
            lp = _container_get_data(context, builder, tl, l)
            base_ptr = builder.call(fn, [lp])
            llty = context.get_data_type(tl.item_type)
            casted_base_ptr = builder.bitcast(base_ptr, llty.as_pointer())
            item_ptr = cgutils.gep(builder, casted_base_ptr, index)
            if is_none:
                out = builder.load(item_ptr)
            else:
                out = context.make_optional_none(builder, tl.item_type)
                pout = cgutils.alloca_once_value(builder, out)
                dm_item = context.data_model_manager[tl.item_type]
                item = dm_item.load_from_data_pointer(builder, item_ptr)
                if not borrowed:
                    context.nrt.incref(builder, tl.item_type, item)
                if is_none:
                    loaded = item
                else:
                    loaded = context.make_optional_value(builder, tl.item_type, item)
                builder.store(loaded, pout)
                out = builder.load(pout)
            return context.make_tuple(builder, resty, [ll_status(0), out])
        return (sig, codegen)
    return impl