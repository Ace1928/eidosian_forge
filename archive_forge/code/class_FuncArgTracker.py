from collections import defaultdict
from sympy.core import Basic, Mul, Add, Pow, sympify
from sympy.core.containers import Tuple, OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol
from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
from sympy.matrices.expressions import (MatrixExpr, MatrixSymbol, MatMul,
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.polys.rootoftools import RootOf
from sympy.utilities.iterables import numbered_symbols, sift, \
from . import cse_opts
class FuncArgTracker:
    """
    A class which manages a mapping from functions to arguments and an inverse
    mapping from arguments to functions.
    """

    def __init__(self, funcs):
        self.value_numbers = {}
        self.value_number_to_value = []
        self.arg_to_funcset = []
        self.func_to_argset = []
        for func_i, func in enumerate(funcs):
            func_argset = OrderedSet()
            for func_arg in func.args:
                arg_number = self.get_or_add_value_number(func_arg)
                func_argset.add(arg_number)
                self.arg_to_funcset[arg_number].add(func_i)
            self.func_to_argset.append(func_argset)

    def get_args_in_value_order(self, argset):
        """
        Return the list of arguments in sorted order according to their value
        numbers.
        """
        return [self.value_number_to_value[argn] for argn in sorted(argset)]

    def get_or_add_value_number(self, value):
        """
        Return the value number for the given argument.
        """
        nvalues = len(self.value_numbers)
        value_number = self.value_numbers.setdefault(value, nvalues)
        if value_number == nvalues:
            self.value_number_to_value.append(value)
            self.arg_to_funcset.append(OrderedSet())
        return value_number

    def stop_arg_tracking(self, func_i):
        """
        Remove the function func_i from the argument to function mapping.
        """
        for arg in self.func_to_argset[func_i]:
            self.arg_to_funcset[arg].remove(func_i)

    def get_common_arg_candidates(self, argset, min_func_i=0):
        """Return a dict whose keys are function numbers. The entries of the dict are
        the number of arguments said function has in common with
        ``argset``. Entries have at least 2 items in common.  All keys have
        value at least ``min_func_i``.
        """
        count_map = defaultdict(lambda: 0)
        if not argset:
            return count_map
        funcsets = [self.arg_to_funcset[arg] for arg in argset]
        largest_funcset = max(funcsets, key=len)
        for funcset in funcsets:
            if largest_funcset is funcset:
                continue
            for func_i in funcset:
                if func_i >= min_func_i:
                    count_map[func_i] += 1
        smaller_funcs_container, larger_funcs_container = sorted([largest_funcset, count_map], key=len)
        for func_i in smaller_funcs_container:
            if count_map[func_i] < 1:
                continue
            if func_i in larger_funcs_container:
                count_map[func_i] += 1
        return {k: v for k, v in count_map.items() if v >= 2}

    def get_subset_candidates(self, argset, restrict_to_funcset=None):
        """
        Return a set of functions each of which whose argument list contains
        ``argset``, optionally filtered only to contain functions in
        ``restrict_to_funcset``.
        """
        iarg = iter(argset)
        indices = OrderedSet((fi for fi in self.arg_to_funcset[next(iarg)]))
        if restrict_to_funcset is not None:
            indices &= restrict_to_funcset
        for arg in iarg:
            indices &= self.arg_to_funcset[arg]
        return indices

    def update_func_argset(self, func_i, new_argset):
        """
        Update a function with a new set of arguments.
        """
        new_args = OrderedSet(new_argset)
        old_args = self.func_to_argset[func_i]
        for deleted_arg in old_args - new_args:
            self.arg_to_funcset[deleted_arg].remove(func_i)
        for added_arg in new_args - old_args:
            self.arg_to_funcset[added_arg].add(func_i)
        self.func_to_argset[func_i].clear()
        self.func_to_argset[func_i].update(new_args)