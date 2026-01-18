from __future__ import annotations
import operator
from typing import Any
import numpy as np
from pandas._libs import lib
from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op
from pandas.core.dtypes.generic import ABCNDFrame
from pandas.core import roperator
from pandas.core.construction import extract_array
from pandas.core.ops.common import unpack_zerodim_and_defer
class OpsMixin:

    def _cmp_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer('__eq__')
    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    @unpack_zerodim_and_defer('__ne__')
    def __ne__(self, other):
        return self._cmp_method(other, operator.ne)

    @unpack_zerodim_and_defer('__lt__')
    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    @unpack_zerodim_and_defer('__le__')
    def __le__(self, other):
        return self._cmp_method(other, operator.le)

    @unpack_zerodim_and_defer('__gt__')
    def __gt__(self, other):
        return self._cmp_method(other, operator.gt)

    @unpack_zerodim_and_defer('__ge__')
    def __ge__(self, other):
        return self._cmp_method(other, operator.ge)

    def _logical_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer('__and__')
    def __and__(self, other):
        return self._logical_method(other, operator.and_)

    @unpack_zerodim_and_defer('__rand__')
    def __rand__(self, other):
        return self._logical_method(other, roperator.rand_)

    @unpack_zerodim_and_defer('__or__')
    def __or__(self, other):
        return self._logical_method(other, operator.or_)

    @unpack_zerodim_and_defer('__ror__')
    def __ror__(self, other):
        return self._logical_method(other, roperator.ror_)

    @unpack_zerodim_and_defer('__xor__')
    def __xor__(self, other):
        return self._logical_method(other, operator.xor)

    @unpack_zerodim_and_defer('__rxor__')
    def __rxor__(self, other):
        return self._logical_method(other, roperator.rxor)

    def _arith_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer('__add__')
    def __add__(self, other):
        """
        Get Addition of DataFrame and other, column-wise.

        Equivalent to ``DataFrame.add(other)``.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Object to be added to the DataFrame.

        Returns
        -------
        DataFrame
            The result of adding ``other`` to DataFrame.

        See Also
        --------
        DataFrame.add : Add a DataFrame and another object, with option for index-
            or column-oriented addition.

        Examples
        --------
        >>> df = pd.DataFrame({'height': [1.5, 2.6], 'weight': [500, 800]},
        ...                   index=['elk', 'moose'])
        >>> df
               height  weight
        elk       1.5     500
        moose     2.6     800

        Adding a scalar affects all rows and columns.

        >>> df[['height', 'weight']] + 1.5
               height  weight
        elk       3.0   501.5
        moose     4.1   801.5

        Each element of a list is added to a column of the DataFrame, in order.

        >>> df[['height', 'weight']] + [0.5, 1.5]
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        Keys of a dictionary are aligned to the DataFrame, based on column names;
        each value in the dictionary is added to the corresponding column.

        >>> df[['height', 'weight']] + {'height': 0.5, 'weight': 1.5}
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        When `other` is a :class:`Series`, the index of `other` is aligned with the
        columns of the DataFrame.

        >>> s1 = pd.Series([0.5, 1.5], index=['weight', 'height'])
        >>> df[['height', 'weight']] + s1
               height  weight
        elk       3.0   500.5
        moose     4.1   800.5

        Even when the index of `other` is the same as the index of the DataFrame,
        the :class:`Series` will not be reoriented. If index-wise alignment is desired,
        :meth:`DataFrame.add` should be used with `axis='index'`.

        >>> s2 = pd.Series([0.5, 1.5], index=['elk', 'moose'])
        >>> df[['height', 'weight']] + s2
               elk  height  moose  weight
        elk    NaN     NaN    NaN     NaN
        moose  NaN     NaN    NaN     NaN

        >>> df[['height', 'weight']].add(s2, axis='index')
               height  weight
        elk       2.0   500.5
        moose     4.1   801.5

        When `other` is a :class:`DataFrame`, both columns names and the
        index are aligned.

        >>> other = pd.DataFrame({'height': [0.2, 0.4, 0.6]},
        ...                      index=['elk', 'moose', 'deer'])
        >>> df[['height', 'weight']] + other
               height  weight
        deer      NaN     NaN
        elk       1.7     NaN
        moose     3.0     NaN
        """
        return self._arith_method(other, operator.add)

    @unpack_zerodim_and_defer('__radd__')
    def __radd__(self, other):
        return self._arith_method(other, roperator.radd)

    @unpack_zerodim_and_defer('__sub__')
    def __sub__(self, other):
        return self._arith_method(other, operator.sub)

    @unpack_zerodim_and_defer('__rsub__')
    def __rsub__(self, other):
        return self._arith_method(other, roperator.rsub)

    @unpack_zerodim_and_defer('__mul__')
    def __mul__(self, other):
        return self._arith_method(other, operator.mul)

    @unpack_zerodim_and_defer('__rmul__')
    def __rmul__(self, other):
        return self._arith_method(other, roperator.rmul)

    @unpack_zerodim_and_defer('__truediv__')
    def __truediv__(self, other):
        return self._arith_method(other, operator.truediv)

    @unpack_zerodim_and_defer('__rtruediv__')
    def __rtruediv__(self, other):
        return self._arith_method(other, roperator.rtruediv)

    @unpack_zerodim_and_defer('__floordiv__')
    def __floordiv__(self, other):
        return self._arith_method(other, operator.floordiv)

    @unpack_zerodim_and_defer('__rfloordiv')
    def __rfloordiv__(self, other):
        return self._arith_method(other, roperator.rfloordiv)

    @unpack_zerodim_and_defer('__mod__')
    def __mod__(self, other):
        return self._arith_method(other, operator.mod)

    @unpack_zerodim_and_defer('__rmod__')
    def __rmod__(self, other):
        return self._arith_method(other, roperator.rmod)

    @unpack_zerodim_and_defer('__divmod__')
    def __divmod__(self, other):
        return self._arith_method(other, divmod)

    @unpack_zerodim_and_defer('__rdivmod__')
    def __rdivmod__(self, other):
        return self._arith_method(other, roperator.rdivmod)

    @unpack_zerodim_and_defer('__pow__')
    def __pow__(self, other):
        return self._arith_method(other, operator.pow)

    @unpack_zerodim_and_defer('__rpow__')
    def __rpow__(self, other):
        return self._arith_method(other, roperator.rpow)