from numba.extending import typeof_impl
from numba.extending import models, register_model
from numba.extending import unbox, NativeValue, box
from numba.core.imputils import lower_constant, lower_cast
from numba.core.ccallback import CFunc
from numba.core import cgutils
from llvmlite import ir
from numba.core import types
from numba.core.types import (FunctionType, UndefinedFunctionType,
from numba.core.dispatcher import Dispatcher
@register_model(FunctionPrototype)
class FunctionProtoModel(models.PrimitiveModel):
    """FunctionProtoModel describes the signatures of first-class functions
    """

    def __init__(self, dmm, fe_type):
        if isinstance(fe_type, FunctionType):
            ftype = fe_type.ftype
        elif isinstance(fe_type, FunctionPrototype):
            ftype = fe_type
        else:
            raise NotImplementedError(type(fe_type))
        retty = dmm.lookup(ftype.rtype).get_value_type()
        args = [dmm.lookup(t).get_value_type() for t in ftype.atypes]
        be_type = ir.PointerType(ir.FunctionType(retty, args))
        super(FunctionProtoModel, self).__init__(dmm, fe_type, be_type)