from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
class ArrayPrinter:

    def _arrayify(self, indexed):
        from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
        try:
            return convert_indexed_to_array(indexed)
        except Exception:
            return indexed

    def _get_einsum_string(self, subranks, contraction_indices):
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ''
        counter = 0
        d = {j: min(i) for i in contraction_indices for j in i}
        indices = []
        for rank_arg in subranks:
            lindices = []
            for i in range(rank_arg):
                if counter in d:
                    lindices.append(d[counter])
                else:
                    lindices.append(counter)
                counter += 1
            indices.append(lindices)
        mapping = {}
        letters_free = []
        letters_dum = []
        for i in indices:
            for j in i:
                if j not in mapping:
                    l = next(letters)
                    mapping[j] = l
                else:
                    l = mapping[j]
                contraction_string += l
                if j in d:
                    if l not in letters_dum:
                        letters_dum.append(l)
                else:
                    letters_free.append(l)
            contraction_string += ','
        contraction_string = contraction_string[:-1]
        return (contraction_string, letters_free, letters_dum)

    def _get_letter_generator_for_einsum(self):
        for i in range(97, 123):
            yield chr(i)
        for i in range(65, 91):
            yield chr(i)
        raise ValueError('out of letters')

    def _print_ArrayTensorProduct(self, expr):
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ','.join([''.join([next(letters) for j in range(i)]) for i in expr.subranks])
        return '%s("%s", %s)' % (self._module_format(self._module + '.' + self._einsum), contraction_string, ', '.join([self._print(arg) for arg in expr.args]))

    def _print_ArrayContraction(self, expr):
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        base = expr.expr
        contraction_indices = expr.contraction_indices
        if isinstance(base, ArrayTensorProduct):
            elems = ','.join(['%s' % self._print(arg) for arg in base.args])
            ranks = base.subranks
        else:
            elems = self._print(base)
            ranks = [len(base.shape)]
        contraction_string, letters_free, letters_dum = self._get_einsum_string(ranks, contraction_indices)
        if not contraction_indices:
            return self._print(base)
        if isinstance(base, ArrayTensorProduct):
            elems = ','.join(['%s' % self._print(arg) for arg in base.args])
        else:
            elems = self._print(base)
        return '%s("%s", %s)' % (self._module_format(self._module + '.' + self._einsum), '{}->{}'.format(contraction_string, ''.join(sorted(letters_free))), elems)

    def _print_ArrayDiagonal(self, expr):
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        diagonal_indices = list(expr.diagonal_indices)
        if isinstance(expr.expr, ArrayTensorProduct):
            subranks = expr.expr.subranks
            elems = expr.expr.args
        else:
            subranks = expr.subranks
            elems = [expr.expr]
        diagonal_string, letters_free, letters_dum = self._get_einsum_string(subranks, diagonal_indices)
        elems = [self._print(i) for i in elems]
        return '%s("%s", %s)' % (self._module_format(self._module + '.' + self._einsum), '{}->{}'.format(diagonal_string, ''.join(letters_free + letters_dum)), ', '.join(elems))

    def _print_PermuteDims(self, expr):
        return '%s(%s, %s)' % (self._module_format(self._module + '.' + self._transpose), self._print(expr.expr), self._print(expr.permutation.array_form))

    def _print_ArrayAdd(self, expr):
        return self._expand_fold_binary_op(self._module + '.' + self._add, expr.args)

    def _print_OneArray(self, expr):
        return '%s((%s,))' % (self._module_format(self._module + '.' + self._ones), ','.join(map(self._print, expr.args)))

    def _print_ZeroArray(self, expr):
        return '%s((%s,))' % (self._module_format(self._module + '.' + self._zeros), ','.join(map(self._print, expr.args)))

    def _print_Assignment(self, expr):
        lhs = self._print(self._arrayify(expr.lhs))
        rhs = self._print(self._arrayify(expr.rhs))
        return '%s = %s' % (lhs, rhs)

    def _print_IndexedBase(self, expr):
        return self._print_ArraySymbol(expr)