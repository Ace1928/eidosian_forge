from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class DatatypeSortRef(SortRef):
    """Datatype sorts."""

    def num_constructors(self):
        """Return the number of constructors in the given Z3 datatype.

        >>> List = Datatype('List')
        >>> List.declare('cons', ('car', IntSort()), ('cdr', List))
        >>> List.declare('nil')
        >>> List = List.create()
        >>> # List is now a Z3 declaration
        >>> List.num_constructors()
        2
        """
        return int(Z3_get_datatype_sort_num_constructors(self.ctx_ref(), self.ast))

    def constructor(self, idx):
        """Return a constructor of the datatype `self`.

        >>> List = Datatype('List')
        >>> List.declare('cons', ('car', IntSort()), ('cdr', List))
        >>> List.declare('nil')
        >>> List = List.create()
        >>> # List is now a Z3 declaration
        >>> List.num_constructors()
        2
        >>> List.constructor(0)
        cons
        >>> List.constructor(1)
        nil
        """
        if z3_debug():
            _z3_assert(idx < self.num_constructors(), 'Invalid constructor index')
        return FuncDeclRef(Z3_get_datatype_sort_constructor(self.ctx_ref(), self.ast, idx), self.ctx)

    def recognizer(self, idx):
        """In Z3, each constructor has an associated recognizer predicate.

        If the constructor is named `name`, then the recognizer `is_name`.

        >>> List = Datatype('List')
        >>> List.declare('cons', ('car', IntSort()), ('cdr', List))
        >>> List.declare('nil')
        >>> List = List.create()
        >>> # List is now a Z3 declaration
        >>> List.num_constructors()
        2
        >>> List.recognizer(0)
        is(cons)
        >>> List.recognizer(1)
        is(nil)
        >>> simplify(List.is_nil(List.cons(10, List.nil)))
        False
        >>> simplify(List.is_cons(List.cons(10, List.nil)))
        True
        >>> l = Const('l', List)
        >>> simplify(List.is_cons(l))
        is(cons, l)
        """
        if z3_debug():
            _z3_assert(idx < self.num_constructors(), 'Invalid recognizer index')
        return FuncDeclRef(Z3_get_datatype_sort_recognizer(self.ctx_ref(), self.ast, idx), self.ctx)

    def accessor(self, i, j):
        """In Z3, each constructor has 0 or more accessor.
        The number of accessors is equal to the arity of the constructor.

        >>> List = Datatype('List')
        >>> List.declare('cons', ('car', IntSort()), ('cdr', List))
        >>> List.declare('nil')
        >>> List = List.create()
        >>> List.num_constructors()
        2
        >>> List.constructor(0)
        cons
        >>> num_accs = List.constructor(0).arity()
        >>> num_accs
        2
        >>> List.accessor(0, 0)
        car
        >>> List.accessor(0, 1)
        cdr
        >>> List.constructor(1)
        nil
        >>> num_accs = List.constructor(1).arity()
        >>> num_accs
        0
        """
        if z3_debug():
            _z3_assert(i < self.num_constructors(), 'Invalid constructor index')
            _z3_assert(j < self.constructor(i).arity(), 'Invalid accessor index')
        return FuncDeclRef(Z3_get_datatype_sort_constructor_accessor(self.ctx_ref(), self.ast, i, j), ctx=self.ctx)