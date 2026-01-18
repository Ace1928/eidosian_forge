from sympy.core.expr import ExprBuilder
from sympy.core.function import (Function, FunctionClass, Lambda)
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.matrices.matrices import MatrixBase
def _eval_derivative_matrix_lines(self, x):
    from sympy.matrices.expressions.special import Identity
    from sympy.tensor.array.expressions.array_expressions import ArrayContraction
    from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
    from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
    fdiff = self._get_function_fdiff()
    lr = self.expr._eval_derivative_matrix_lines(x)
    ewdiff = ElementwiseApplyFunction(fdiff, self.expr)
    if 1 in x.shape:
        iscolumn = self.shape[1] == 1
        for i in lr:
            if iscolumn:
                ptr1 = i.first_pointer
                ptr2 = Identity(self.shape[1])
            else:
                ptr1 = Identity(self.shape[0])
                ptr2 = i.second_pointer
            subexpr = ExprBuilder(ArrayDiagonal, [ExprBuilder(ArrayTensorProduct, [ewdiff, ptr1, ptr2]), (0, 2) if iscolumn else (1, 4)], validator=ArrayDiagonal._validate)
            i._lines = [subexpr]
            i._first_pointer_parent = subexpr.args[0].args
            i._first_pointer_index = 1
            i._second_pointer_parent = subexpr.args[0].args
            i._second_pointer_index = 2
    else:
        for i in lr:
            ptr1 = i.first_pointer
            ptr2 = i.second_pointer
            newptr1 = Identity(ptr1.shape[1])
            newptr2 = Identity(ptr2.shape[1])
            subexpr = ExprBuilder(ArrayContraction, [ExprBuilder(ArrayTensorProduct, [ptr1, newptr1, ewdiff, ptr2, newptr2]), (1, 2, 4), (5, 7, 8)], validator=ArrayContraction._validate)
            i._first_pointer_parent = subexpr.args[0].args
            i._first_pointer_index = 1
            i._second_pointer_parent = subexpr.args[0].args
            i._second_pointer_index = 4
            i._lines = [subexpr]
    return lr