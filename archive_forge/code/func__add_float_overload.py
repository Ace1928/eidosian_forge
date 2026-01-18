import operator
from numba import njit, literally
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError
from numba.core.extending import lower_builtin
from numba.core.extending import models, register_model
from numba.core.extending import make_attribute_wrapper
from numba.core.extending import type_callable
from numba.core.extending import overload
from numba.core.extending import typeof_impl
import unittest
def _add_float_overload(self, mock_float_inst):

    @overload(mock_float_inst)
    def dummy_to_float(x):
        if isinstance(x, self.DummyType):

            def codegen(x):
                return float(x.value)
            return codegen
        else:
            raise NumbaTypeError('cannot type float({})'.format(x))