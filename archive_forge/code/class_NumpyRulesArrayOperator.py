import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
class NumpyRulesArrayOperator(Numpy_rules_ufunc):
    _op_map = {operator.add: 'add', operator.sub: 'subtract', operator.mul: 'multiply', operator.truediv: 'true_divide', operator.floordiv: 'floor_divide', operator.mod: 'remainder', operator.pow: 'power', operator.lshift: 'left_shift', operator.rshift: 'right_shift', operator.and_: 'bitwise_and', operator.or_: 'bitwise_or', operator.xor: 'bitwise_xor', operator.eq: 'equal', operator.gt: 'greater', operator.ge: 'greater_equal', operator.lt: 'less', operator.le: 'less_equal', operator.ne: 'not_equal'}

    @property
    def ufunc(self):
        return getattr(np, self._op_map[self.key])

    @classmethod
    def install_operations(cls):
        for op, ufunc_name in cls._op_map.items():
            infer_global(op)(type('NumpyRulesArrayOperator_' + ufunc_name, (cls,), dict(key=op)))

    def generic(self, args, kws):
        """Overloads and calls base class generic() method, returning
        None if a TypingError occurred.

        Returning None for operators is important since operators are
        heavily overloaded, and by suppressing type errors, we allow
        type inference to check other possibilities before giving up
        (particularly user-defined operators).
        """
        try:
            sig = super(NumpyRulesArrayOperator, self).generic(args, kws)
        except TypingError:
            return None
        if sig is None:
            return None
        args = sig.args
        if not any((isinstance(arg, types.ArrayCompatible) for arg in args)):
            return None
        return sig