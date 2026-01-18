from __future__ import annotations
import warnings
from typing import List, Tuple, TypeVar
import numpy as np
from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
class OpRelEntrConeQuad(Cone):
    """An approximate construction of the operator relative entropy cone

    Definition:

    .. math::

        K_{re}^n=\\text{cl}\\{(X,Y,T)\\in\\mathbb{H}^n_{++}\\times
                \\mathbb{H}^n_{++}\\times\\mathbb{H}^n_{++}\\:D_{\\text{op}}\\succeq T\\}

    More details on the approximation can be found in Theorem-3 on page-10 in the paper:
    Semidefinite Approximations of the Matrix Logarithm.

    Parameters
    ----------
    X : Expression
        x in the (approximate) operator relative entropy cone
    Y : Expression
        y in the (approximate) operator relative entropy cone
    Z : Expression
        Z in the (approximate) operator relative entropy cone
    m: int
        Must be positive. Controls the number of quadrature nodes used in a local
        approximation of the matrix logarithm. Increasing this value results in
        better local approximations, but does not significantly expand the region
        of inputs for which the approximation is effective.
    k: int
        Must be positive. Sets the number of scaling points about which the
        quadrature approximation is performed. Increasing this value will
        expand the region of inputs over which the approximation is effective.

    This approximation uses :math:`m + k` semidefinite constraints.
    """

    def __init__(self, X: Expression, Y: Expression, Z: Expression, m: int, k: int, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        self.X = Expression.cast_to_const(X)
        self.Y = Expression.cast_to_const(Y)
        self.Z = Expression.cast_to_const(Z)
        if not X.is_hermitian() or not Y.is_hermitian() or (not Z.is_hermitian()):
            msg = 'One of the input matrices has not explicitly been declared as symmetric orHermitian. If the inputs are Variable objects, try declaring them with thesymmetric=True or Hermitian=True properties. If the inputs are general Expression objects that are known to be symmetric or Hermitian, then youcan wrap them with the symmetric_wrap and hermitian_wrap atoms. Failure todo one of these things will cause this function to impose a symmetry orconjugate-symmetry constraint internally, in a way that is veryinefficient.'
            warnings.warn(msg)
        self.m = m
        self.k = k
        Xs, Ys, Zs = (self.X.shape, self.Y.shape, self.Z.shape)
        if Xs != Ys or Xs != Zs:
            msg = 'All arguments must have the same shapes. Provided arguments haveshapes %s' % str((Xs, Ys, Zs))
            raise ValueError(msg)
        super(OpRelEntrConeQuad, self).__init__([self.X, self.Y, self.Z], constr_id)

    def get_data(self):
        return [self.m, self.k, self.id]

    def __str__(self) -> str:
        tup = (self.X, self.Y, self.Z, self.m, self.k)
        return 'OpRelEntrConeQuad(%s, %s, %s, %s, %s)' % tup

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def residual(self):
        raise NotImplementedError()

    @property
    def size(self) -> int:
        """The number of entries in the combined cones.
        """
        return 3 * self.num_cones()

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.X.size

    def cone_sizes(self) -> List[int]:
        """The dimensions of the exponential cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        return [3] * self.num_cones()

    def is_dcp(self, dpp: bool=False) -> bool:
        """An operator relative conic constraint is DCP when (A, b, C) is affine
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
        s = (3,) + self.X.shape
        return s

    def save_dual_value(self, value) -> None:
        pass