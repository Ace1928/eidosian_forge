from __future__ import annotations
import warnings
from typing import List, Tuple, TypeVar
import numpy as np
from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
class ExpCone(Cone):
    """A reformulated exponential cone constraint.

    Operates elementwise on :math:`x, y, z`.

    Original cone:

    .. math::

        K = \\{(x,y,z) \\mid y > 0, ye^{x/y} <= z\\}
            \\cup \\{(x,y,z) \\mid x \\leq 0, y = 0, z \\geq 0\\}

    Reformulated cone:

    .. math::

        K = \\{(x,y,z) \\mid y, z > 0, y\\log(y) + x \\leq y\\log(z)\\}
             \\cup \\{(x,y,z) \\mid x \\leq 0, y = 0, z \\geq 0\\}

    Parameters
    ----------
    x : Expression
        x in the exponential cone.
    y : Expression
        y in the exponential cone.
    z : Expression
        z in the exponential cone.
    """

    def __init__(self, x: Expression, y: Expression, z: Expression, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        self.x = Expression.cast_to_const(x)
        self.y = Expression.cast_to_const(y)
        self.z = Expression.cast_to_const(z)
        args = [self.x, self.y, self.z]
        for val in args:
            if not (val.is_affine() and val.is_real()):
                raise ValueError('All arguments must be affine and real.')
        xs, ys, zs = (self.x.shape, self.y.shape, self.z.shape)
        if xs != ys or xs != zs:
            msg = 'All arguments must have the same shapes. Provided arguments haveshapes %s' % str((xs, ys, zs))
            raise ValueError(msg)
        super(ExpCone, self).__init__(args, constr_id)

    def __str__(self) -> str:
        return 'ExpCone(%s, %s, %s)' % (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return 'ExpCone(%s, %s, %s)' % (self.x, self.y, self.z)

    @property
    def residual(self):
        from cvxpy import Minimize, Problem, Variable, hstack, norm2
        if self.x.value is None or self.y.value is None or self.z.value is None:
            return None
        x = Variable(self.x.shape)
        y = Variable(self.y.shape)
        z = Variable(self.z.shape)
        constr = [ExpCone(x, y, z)]
        obj = Minimize(norm2(hstack([x, y, z]) - hstack([self.x.value, self.y.value, self.z.value])))
        problem = Problem(obj, constr)
        return problem.solve()

    @property
    def size(self) -> int:
        """The number of entries in the combined cones.
        """
        return 3 * self.num_cones()

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.x.size

    def as_quad_approx(self, m: int, k: int) -> RelEntrConeQuad:
        return RelEntrConeQuad(self.y, self.z, -self.x, m, k)

    def cone_sizes(self) -> List[int]:
        """The dimensions of the exponential cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        return [3] * self.num_cones()

    def is_dcp(self, dpp: bool=False) -> bool:
        """An exponential constraint is DCP if each argument is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                return all((arg.is_affine() for arg in self.args))
        return all((arg.is_affine() for arg in self.args))

    def is_dgp(self, dpp: bool=False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    @property
    def shape(self) -> Tuple[int, ...]:
        s = (3,) + self.x.shape
        return s

    def save_dual_value(self, value) -> None:
        value = np.reshape(value, newshape=(-1, 3))
        dv0 = np.reshape(value[:, 0], newshape=self.x.shape)
        dv1 = np.reshape(value[:, 1], newshape=self.y.shape)
        dv2 = np.reshape(value[:, 2], newshape=self.z.shape)
        self.dual_variables[0].save_value(dv0)
        self.dual_variables[1].save_value(dv1)
        self.dual_variables[2].save_value(dv2)

    def _dual_cone(self, *args):
        """Implements the dual cone of the exponential cone
        See Pg 85 of the MOSEK modelling cookbook for more information"""
        if args == ():
            return ExpCone(-self.dual_variables[1], -self.dual_variables[0], np.exp(1) * self.dual_variables[2])
        else:
            args_shapes = [arg.shape for arg in args]
            instance_args_shapes = [arg.shape for arg in self.args]
            assert len(args) == len(self.args)
            assert args_shapes == instance_args_shapes
            return ExpCone(-args[1], -args[0], np.exp(1) * args[2])