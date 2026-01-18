from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.printing.defaults import Printable
import itertools
from collections.abc import Iterable
class ArrayKind(Kind):
    """
    Kind for N-dimensional array in SymPy.

    This kind represents the multidimensional array that algebraic
    operations are defined. Basic class for this kind is ``NDimArray``,
    but any expression representing the array can have this.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is :obj:NumberKind `<sympy.core.kind.NumberKind>`,
        which means that the array contains only numbers.

    Examples
    ========

    Any instance of array class has ``ArrayKind``.

    >>> from sympy import NDimArray
    >>> NDimArray([1,2,3]).kind
    ArrayKind(NumberKind)

    Although expressions representing an array may be not instance of
    array class, it will have ``ArrayKind`` as well.

    >>> from sympy import Integral
    >>> from sympy.tensor.array import NDimArray
    >>> from sympy.abc import x
    >>> intA = Integral(NDimArray([1,2,3]), x)
    >>> isinstance(intA, NDimArray)
    False
    >>> intA.kind
    ArrayKind(NumberKind)

    Use ``isinstance()`` to check for ``ArrayKind` without specifying
    the element kind. Use ``is`` with specifying the element kind.

    >>> from sympy.tensor.array import ArrayKind
    >>> from sympy.core import NumberKind
    >>> boolA = NDimArray([True, False])
    >>> isinstance(boolA.kind, ArrayKind)
    True
    >>> boolA.kind is ArrayKind(NumberKind)
    False

    See Also
    ========

    shape : Function to return the shape of objects with ``MatrixKind``.

    """

    def __new__(cls, element_kind=NumberKind):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        return 'ArrayKind(%s)' % self.element_kind

    @classmethod
    def _union(cls, kinds) -> 'ArrayKind':
        elem_kinds = {e.kind for e in kinds}
        if len(elem_kinds) == 1:
            elemkind, = elem_kinds
        else:
            elemkind = UndefinedKind
        return ArrayKind(elemkind)