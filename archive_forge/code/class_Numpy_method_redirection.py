import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
class Numpy_method_redirection(AbstractTemplate):
    """
    A template redirecting a Numpy global function (e.g. np.sum) to an
    array method of the same name (e.g. ndarray.sum).
    """
    prefer_literal = True

    def generic(self, args, kws):
        pysig = None
        if kws:
            if self.method_name == 'sum':
                if 'axis' in kws and 'dtype' not in kws:

                    def sum_stub(arr, axis):
                        pass
                    pysig = utils.pysignature(sum_stub)
                elif 'dtype' in kws and 'axis' not in kws:

                    def sum_stub(arr, dtype):
                        pass
                    pysig = utils.pysignature(sum_stub)
                elif 'dtype' in kws and 'axis' in kws:

                    def sum_stub(arr, axis, dtype):
                        pass
                    pysig = utils.pysignature(sum_stub)
            elif self.method_name == 'argsort':

                def argsort_stub(arr, kind='quicksort'):
                    pass
                pysig = utils.pysignature(argsort_stub)
            else:
                fmt = "numba doesn't support kwarg for {}"
                raise TypingError(fmt.format(self.method_name))
        arr = args[0]
        meth_ty = self.context.resolve_getattr(arr, self.method_name)
        meth_sig = self.context.resolve_function_type(meth_ty, args[1:], kws)
        if meth_sig is not None:
            return meth_sig.as_function().replace(pysig=pysig)