from typing import List, Tuple
import numpy as np
from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
class PowConeND(Cone):
    """
    Represents a collection of N-dimensional power cone constraints
    that is *mathematically* equivalent to the following code
    snippet (which makes incorrect use of numpy functions on cvxpy
    objects):

        np.prod(np.power(W, alpha), axis=axis) >= np.abs(z),
        W >= 0

    All arguments must be Expression-like, and z must satisfy
    z.ndim <= 1. The columns (rows) of alpha must sum to 1 when
    axis=0 (axis=1).

    Note: unlike PowCone3D, we make no attempt to promote
    alpha to the appropriate shape. The dimensions of W and
    alpha must match exactly.

    Note: Dual variables are not currently implemented for this type
    of constraint.
    """
    _TOL_ = 1e-06

    def __init__(self, W, z, alpha, axis: int=0, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        W = Expression.cast_to_const(W)
        if not (W.is_real() and W.is_affine()):
            msg = 'Invalid first argument; W must be affine and real.'
            raise ValueError(msg)
        z = Expression.cast_to_const(z)
        if z.ndim > 1 or not (z.is_real() and z.is_affine()):
            msg = 'Invalid second argument. z must be affine, real, and have at most one z.ndim <= 1.'
            raise ValueError(msg)
        if W.ndim <= 1 and z.size > 1 or (W.ndim == 2 and z.size != W.shape[1 - axis]) or (W.ndim == 1 and axis == 1):
            raise ValueError('Argument dimensions %s and %s, with axis=%i, are incompatible.' % (W.shape, z.shape, axis))
        if W.ndim == 2 and W.shape[axis] <= 1:
            msg = 'PowConeND requires left-hand-side to have at least two terms.'
            raise ValueError(msg)
        alpha = Expression.cast_to_const(alpha)
        if alpha.shape != W.shape:
            raise ValueError('Argument dimensions %s and %s are not equal.' % (W.shape, alpha.shape))
        if np.any(alpha.value <= 0):
            raise ValueError('Argument alpha must be entry-wise positive.')
        if np.any(np.abs(1 - np.sum(alpha.value, axis=axis)) > PowConeND._TOL_):
            raise ValueError('Argument alpha must sum to 1 along axis %s.' % axis)
        self.W = W
        self.z = z
        self.alpha = alpha
        self.axis = axis
        if z.ndim == 0:
            z = z.flatten()
        super(PowConeND, self).__init__([W, z], constr_id)

    def __str__(self) -> str:
        return 'PowND(%s, %s; %s)' % (self.W, self.z, self.alpha)

    def is_imag(self) -> bool:
        return False

    def is_complex(self) -> bool:
        return False

    def get_data(self):
        return [self.alpha, self.axis, self.id]

    @property
    def residual(self):
        from cvxpy import Minimize, Problem, Variable, hstack, norm2
        if self.W.value is None or self.z.value is None:
            return None
        W = Variable(self.W.shape)
        z = Variable(self.z.shape)
        constr = [PowConeND(W, z, self.alpha, axis=self.axis)]
        obj = Minimize(norm2(hstack([W.flatten(), z.flatten()]) - hstack([self.W.flatten().value, self.z.flatten().value])))
        problem = Problem(obj, constr)
        return problem.solve(solver='SCS', eps=1e-08)

    def num_cones(self):
        return self.z.size

    @property
    def size(self) -> int:
        cone_size = 1 + self.args[0].shape[self.axis]
        return cone_size * self.num_cones()

    def cone_sizes(self) -> List[int]:
        cone_size = 1 + self.args[0].shape[self.axis]
        return [cone_size] * self.num_cones()

    def is_dcp(self, dpp: bool=False) -> bool:
        """A power cone constraint is DCP if each argument is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                args_ok = self.args[0].is_affine() and self.args[1].is_affine()
                exps_ok = not isinstance(self.alpha, cvxtypes.parameter())
                return args_ok and exps_ok
        return True

    def is_dgp(self, dpp: bool=False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    def save_dual_value(self, value) -> None:
        dW = value[:, :-1]
        dz = value[:, -1]
        if self.axis == 0:
            dW = dW.T
            dz = dz.T
        if dW.shape[1] == 1:
            dW = np.squeeze(dW)
        self.dual_variables[0].save_value(dW)
        self.dual_variables[1].save_value(dz)

    def _dual_cone(self, *args):
        """Implements the dual cone of PowConeND See Pg 85
        of the MOSEK modelling cookbook for more information"""
        if args is None or args == ():
            scaled_duals = self.dual_variables[0] / self.alpha
            return PowConeND(scaled_duals, self.dual_variables[1], self.alpha, axis=self.axis)
        else:
            args_shapes = [arg.shape for arg in args]
            instance_args_shapes = [arg.shape for arg in self.args]
            assert len(args) == len(self.args)
            assert args_shapes == instance_args_shapes
            assert args[0].value.shape == self.alpha.value.shape
            return PowConeND(args[0] / self.alpha, args[1], self.alpha, axis=self.axis)