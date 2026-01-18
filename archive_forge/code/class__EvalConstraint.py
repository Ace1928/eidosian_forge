from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
class _EvalConstraint(object):

    def __init__(self, variable_names):
        self._variable_names = variable_names
        self._N = len(variable_names)
        self._dispatch = {('VARIABLE', 0): self._eval_variable, ('NUMBER', 0): self._eval_number, ('+', 1): self._eval_unary_plus, ('-', 1): self._eval_unary_minus, ('+', 2): self._eval_binary_plus, ('-', 2): self._eval_binary_minus, ('*', 2): self._eval_binary_multiply, ('/', 2): self._eval_binary_div, ('=', 2): self._eval_binary_eq, (',', 2): self._eval_binary_comma}

    def is_constant(self, coefs):
        return np.all(coefs[:self._N] == 0)

    def _eval_variable(self, tree):
        var = tree.token.extra
        coefs = np.zeros((self._N + 1,), dtype=float)
        coefs[self._variable_names.index(var)] = 1
        return coefs

    def _eval_number(self, tree):
        coefs = np.zeros((self._N + 1,), dtype=float)
        coefs[-1] = float(tree.token.extra)
        return coefs

    def _eval_unary_plus(self, tree):
        return self.eval(tree.args[0])

    def _eval_unary_minus(self, tree):
        return -1 * self.eval(tree.args[0])

    def _eval_binary_plus(self, tree):
        return self.eval(tree.args[0]) + self.eval(tree.args[1])

    def _eval_binary_minus(self, tree):
        return self.eval(tree.args[0]) - self.eval(tree.args[1])

    def _eval_binary_div(self, tree):
        left = self.eval(tree.args[0])
        right = self.eval(tree.args[1])
        if not self.is_constant(right):
            raise PatsyError("Can't divide by a variable in a linear constraint", tree.args[1])
        return left / right[-1]

    def _eval_binary_multiply(self, tree):
        left = self.eval(tree.args[0])
        right = self.eval(tree.args[1])
        if self.is_constant(left):
            return left[-1] * right
        elif self.is_constant(right):
            return left * right[-1]
        else:
            raise PatsyError("Can't multiply one variable by another in a linear constraint", tree)

    def _eval_binary_eq(self, tree):
        args = list(tree.args)
        constraints = []
        for i, arg in enumerate(args):
            if arg.type == '=':
                constraints.append(self.eval(arg, constraint=True))
                args[i] = arg.args[1 - i]
        left = self.eval(args[0])
        right = self.eval(args[1])
        coefs = left[:self._N] - right[:self._N]
        if np.all(coefs == 0):
            raise PatsyError('no variables appear in constraint', tree)
        constant = -left[-1] + right[-1]
        constraint = LinearConstraint(self._variable_names, coefs, constant)
        constraints.append(constraint)
        return LinearConstraint.combine(constraints)

    def _eval_binary_comma(self, tree):
        left = self.eval(tree.args[0], constraint=True)
        right = self.eval(tree.args[1], constraint=True)
        return LinearConstraint.combine([left, right])

    def eval(self, tree, constraint=False):
        key = (tree.type, len(tree.args))
        assert key in self._dispatch
        val = self._dispatch[key](tree)
        if constraint:
            if isinstance(val, LinearConstraint):
                return val
            else:
                assert val.size == self._N + 1
                if np.all(val[:self._N] == 0):
                    raise PatsyError('term is constant, with no variables', tree)
                return LinearConstraint(self._variable_names, val[:self._N], -val[-1])
        else:
            if isinstance(val, LinearConstraint):
                raise PatsyError('unexpected constraint object', tree)
            return val