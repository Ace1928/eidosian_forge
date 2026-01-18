from __future__ import annotations
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas.compat import set_function_name
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.cast import maybe_cast_pointwise_result
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.algorithms import (
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.missing import _fill_limit_area_1d
from pandas.core.sorting import (
class ExtensionScalarOpsMixin(ExtensionOpsMixin):
    """
    A mixin for defining ops on an ExtensionArray.

    It is assumed that the underlying scalar objects have the operators
    already defined.

    Notes
    -----
    If you have defined a subclass MyExtensionArray(ExtensionArray), then
    use MyExtensionArray(ExtensionArray, ExtensionScalarOpsMixin) to
    get the arithmetic operators.  After the definition of MyExtensionArray,
    insert the lines

    MyExtensionArray._add_arithmetic_ops()
    MyExtensionArray._add_comparison_ops()

    to link the operators to your class.

    .. note::

       You may want to set ``__array_priority__`` if you want your
       implementation to be called when involved in binary operations
       with NumPy arrays.
    """

    @classmethod
    def _create_method(cls, op, coerce_to_dtype: bool=True, result_dtype=None):
        """
        A class method that returns a method that will correspond to an
        operator for an ExtensionArray subclass, by dispatching to the
        relevant operator defined on the individual elements of the
        ExtensionArray.

        Parameters
        ----------
        op : function
            An operator that takes arguments op(a, b)
        coerce_to_dtype : bool, default True
            boolean indicating whether to attempt to convert
            the result to the underlying ExtensionArray dtype.
            If it's not possible to create a new ExtensionArray with the
            values, an ndarray is returned instead.

        Returns
        -------
        Callable[[Any, Any], Union[ndarray, ExtensionArray]]
            A method that can be bound to a class. When used, the method
            receives the two arguments, one of which is the instance of
            this class, and should return an ExtensionArray or an ndarray.

            Returning an ndarray may be necessary when the result of the
            `op` cannot be stored in the ExtensionArray. The dtype of the
            ndarray uses NumPy's normal inference rules.

        Examples
        --------
        Given an ExtensionArray subclass called MyExtensionArray, use

            __add__ = cls._create_method(operator.add)

        in the class definition of MyExtensionArray to create the operator
        for addition, that will be based on the operator implementation
        of the underlying elements of the ExtensionArray
        """

        def _binop(self, other):

            def convert_values(param):
                if isinstance(param, ExtensionArray) or is_list_like(param):
                    ovalues = param
                else:
                    ovalues = [param] * len(self)
                return ovalues
            if isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)):
                return NotImplemented
            lvalues = self
            rvalues = convert_values(other)
            res = [op(a, b) for a, b in zip(lvalues, rvalues)]

            def _maybe_convert(arr):
                if coerce_to_dtype:
                    res = maybe_cast_pointwise_result(arr, self.dtype, same_dtype=False)
                    if not isinstance(res, type(self)):
                        res = np.asarray(arr)
                else:
                    res = np.asarray(arr, dtype=result_dtype)
                return res
            if op.__name__ in {'divmod', 'rdivmod'}:
                a, b = zip(*res)
                return (_maybe_convert(a), _maybe_convert(b))
            return _maybe_convert(res)
        op_name = f'__{op.__name__}__'
        return set_function_name(_binop, op_name, cls)

    @classmethod
    def _create_arithmetic_method(cls, op):
        return cls._create_method(op)

    @classmethod
    def _create_comparison_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False, result_dtype=bool)