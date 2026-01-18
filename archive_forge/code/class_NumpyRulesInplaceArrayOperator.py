import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
class NumpyRulesInplaceArrayOperator(NumpyRulesArrayOperator):
    _op_map = {operator.iadd: 'add', operator.isub: 'subtract', operator.imul: 'multiply', operator.itruediv: 'true_divide', operator.ifloordiv: 'floor_divide', operator.imod: 'remainder', operator.ipow: 'power', operator.ilshift: 'left_shift', operator.irshift: 'right_shift', operator.iand: 'bitwise_and', operator.ior: 'bitwise_or', operator.ixor: 'bitwise_xor'}

    def generic(self, args, kws):
        lhs, rhs = args
        if not isinstance(lhs, types.ArrayCompatible):
            return
        args = args + (lhs,)
        sig = super(NumpyRulesInplaceArrayOperator, self).generic(args, kws)
        assert len(sig.args) == 3
        real_sig = signature(sig.return_type, *sig.args[:2])
        return real_sig