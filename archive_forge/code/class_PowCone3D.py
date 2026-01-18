from typing import List, Tuple
import numpy as np
from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
class PowCone3D(Cone):
    """
    An object representing a collection of 3D power cone constraints

        x[i]**alpha[i] * y[i]**(1-alpha[i]) >= |z[i]|  for all i
        x >= 0, y >= 0

    If the parameter alpha is a scalar, it will be promoted to
    a vector matching the (common) sizes of x, y, z. The numeric
    value of alpha (or its components, in the vector case) must
    be a number in the open interval (0, 1).

    We store flattened representations of the arguments (x, y, z,
    and alpha) as Expression objects. We construct dual variables
    with respect to these flattened representations.
    """

    def __init__(self, x, y, z, alpha, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        self.x = Expression.cast_to_const(x)
        self.y = Expression.cast_to_const(y)
        self.z = Expression.cast_to_const(z)
        for val in [self.x, self.y, self.z]:
            if not (val.is_affine() and val.is_real()):
                raise ValueError('All arguments must be affine and real.')
        alpha = Expression.cast_to_const(alpha)
        if alpha.is_scalar():
            if self.x.shape:
                alpha = cvxtypes.promote()(alpha, self.x.shape)
            else:
                alpha = cvxtypes.promote()(alpha, (1,))
        self.alpha = alpha
        if np.any(self.alpha.value <= 0) or np.any(self.alpha.value >= 1):
            msg = 'Argument alpha must have entries in the open interval (0, 1).'
            raise ValueError(msg)
        if alpha.shape == (1,):
            arg_shapes = [self.x.shape, self.y.shape, self.z.shape, ()]
        else:
            arg_shapes = [self.x.shape, self.y.shape, self.z.shape, self.alpha.shape]
        if any((arg_shapes[0] != s for s in arg_shapes[1:])):
            msg = 'All arguments must have the same shapes. Provided arguments haveshapes %s' % str(arg_shapes)
            raise ValueError(msg)
        super(PowCone3D, self).__init__([self.x, self.y, self.z], constr_id)

    def __str__(self) -> str:
        return 'Pow3D(%s, %s, %s; %s)' % (self.x, self.y, self.z, self.alpha)

    @property
    def residual(self):
        from cvxpy import Minimize, Problem, Variable, hstack, norm2
        if self.x.value is None or self.y.value is None or self.z.value is None:
            return None
        x = Variable(self.x.shape)
        y = Variable(self.y.shape)
        z = Variable(self.z.shape)
        constr = [PowCone3D(x, y, z, self.alpha)]
        obj = Minimize(norm2(hstack([x, y, z]) - hstack([self.x.value, self.y.value, self.z.value])))
        problem = Problem(obj, constr)
        return problem.solve(solver='SCS', eps=1e-08)

    def get_data(self):
        return [self.alpha, self.id]

    def is_imag(self) -> bool:
        return False

    def is_complex(self) -> bool:
        return False

    @property
    def size(self) -> int:
        return 3 * self.num_cones()

    def num_cones(self):
        return self.x.size

    def cone_sizes(self) -> List[int]:
        return [3] * self.num_cones()

    def is_dcp(self, dpp: bool=False) -> bool:
        if dpp:
            with scopes.dpp_scope():
                args_ok = all((arg.is_affine() for arg in self.args))
                exps_ok = not isinstance(self.alpha, cvxtypes.parameter())
                return args_ok and exps_ok
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
        value = np.reshape(value, newshape=(3, -1))
        dv0 = np.reshape(value[0, :], newshape=self.x.shape)
        dv1 = np.reshape(value[1, :], newshape=self.y.shape)
        dv2 = np.reshape(value[2, :], newshape=self.z.shape)
        self.dual_variables[0].save_value(dv0)
        self.dual_variables[1].save_value(dv1)
        self.dual_variables[2].save_value(dv2)

    def _dual_cone(self, *args):
        """Implements the dual cone of PowCone3D See Pg 85
        of the MOSEK modelling cookbook for more information"""
        if args is None:
            PowCone3D(self.dual_variables[0] / self.alpha, self.dual_variables[1] / (1 - self.alpha), self.dual_variables[2], self.alpha)
        else:
            args_shapes = [arg.shape for arg in args]
            instance_args_shapes = [arg.shape for arg in self.args]
            assert len(args) == len(self.args)
            assert args_shapes == instance_args_shapes
            return PowCone3D(args[0] / self.alpha, args[1] / (1 - self.alpha), args[2], self.alpha)