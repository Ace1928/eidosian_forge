from typing import List, Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.constants.parameter import is_param_free
class kron(AffAtom):
    """Kronecker product.
    """

    def __init__(self, lh_expr, rh_expr) -> None:
        super(kron, self).__init__(lh_expr, rh_expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Kronecker product of the two values.
        """
        return np.kron(values[0], values[1])

    def validate_arguments(self) -> None:
        """Checks that both arguments are vectors, and the first is constant.
        """
        if not (self.args[0].is_constant() or self.args[1].is_constant()):
            raise ValueError('At least one argument to kron must be constant.')
        elif self.args[0].ndim != 2 or self.args[1].ndim != 2:
            raise ValueError('kron requires matrix arguments.')

    def shape_from_args(self) -> Tuple[int, int]:
        rows = self.args[0].shape[0] * self.args[1].shape[0]
        cols = self.args[0].shape[1] * self.args[1].shape[1]
        return (rows, cols)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        if u.scopes.dpp_scope_active():
            x = self.args[0]
            y = self.args[1]
            return (x.is_constant() or y.is_constant()) and (is_param_free(x) and is_param_free(y))
        else:
            return self.args[0].is_constant() or self.args[1].is_constant()

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return self.is_atom_convex()

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Same as times.
        """
        return u.sign.mul_sign(self.args[0], self.args[1])

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        cst_loc = 0 if self.args[0].is_constant() else 1
        return self.args[cst_loc].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        cst_loc = 0 if self.args[0].is_constant() else 1
        return self.args[cst_loc].is_nonpos()

    def is_psd(self):
        """Check a *sufficient condition* that the expression is PSD,
        by checking if both arguments are PSD or both are NSD.
        """
        case1 = self.args[0].is_psd() and self.args[1].is_psd()
        case2 = self.args[0].is_nsd() and self.args[1].is_nsd()
        return case1 or case2

    def is_nsd(self):
        """Check a *sufficient condition* that the expression is NSD,
        by checking if one argument is PSD and the other is NSD.
        """
        case1 = self.args[0].is_psd() and self.args[1].is_nsd()
        case2 = self.args[0].is_nsd() and self.args[1].is_psd()
        return case1 or case2

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        """Kronecker product of two matrices.

        Parameters
        ----------
        arg_objs : list
            LinOp for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        if self.args[0].is_constant():
            return (lu.kron_r(arg_objs[0], arg_objs[1], shape), [])
        else:
            return (lu.kron_l(arg_objs[0], arg_objs[1], shape), [])