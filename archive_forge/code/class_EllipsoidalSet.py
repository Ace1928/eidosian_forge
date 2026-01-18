import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters
class EllipsoidalSet(UncertaintySet):
    """
    A general ellipsoid.

    Parameters
    ----------
    center : (N,) array-like
        Center of the ellipsoid.
    shape_matrix : (N, N) array-like
        A positive definite matrix characterizing the shape
        and orientation of the ellipsoid.
    scale : numeric type, optional
        Square of the factor by which to scale the semi-axes
        of the ellipsoid (i.e. the eigenvectors of the shape
        matrix). The default is `1`.

    Examples
    --------
    3D origin-centered unit hypersphere:

    >>> from pyomo.contrib.pyros import EllipsoidalSet
    >>> import numpy as np
    >>> hypersphere = EllipsoidalSet(
    ...     center=[0, 0, 0],
    ...     shape_matrix=np.eye(3),
    ...     scale=1,
    ... )
    >>> hypersphere.center
    array([0, 0, 0])
    >>> hypersphere.shape_matrix
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> hypersphere.scale
    1

    A 2D ellipsoid with custom rotation and scaling:

    >>> rotated_ellipsoid = EllipsoidalSet(
    ...     center=[1, 1],
    ...     shape_matrix=[[4, 2], [2, 4]],
    ...     scale=0.5,
    ... )
    >>> rotated_ellipsoid.center
    array([1, 1])
    >>> rotated_ellipsoid.shape_matrix
    array([[4, 2],
           [2, 4]])
    >>> rotated_ellipsoid.scale
    0.5

    """

    def __init__(self, center, shape_matrix, scale=1):
        """Initialize self (see class docstring)."""
        self.center = center
        self.shape_matrix = shape_matrix
        self.scale = scale

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return 'ellipsoidal'

    @property
    def center(self):
        """
        (N,) numpy.ndarray : Center of the ellipsoid.
        """
        return self._center

    @center.setter
    def center(self, val):
        validate_array(arr=val, arr_name='center', dim=1, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
        val_arr = np.array(val)
        if hasattr(self, '_center'):
            if val_arr.size != self.dim:
                raise ValueError(f"Attempting to set attribute 'center' of AxisAlignedEllipsoidalSet of dimension {self.dim} to value of dimension {val_arr.size}")
        self._center = val_arr

    @staticmethod
    def _verify_positive_definite(matrix):
        """
        Verify that a given symmetric square matrix is positive
        definite. An exception is raised if the square matrix
        is not positive definite.

        Parameters
        ----------
        matrix : (N, N) array_like
            Candidate matrix.

        Raises
        ------
        ValueError
            If matrix is not symmetric, not positive definite,
            or the square roots of the diagonal entries are
            not accessible.
        LinAlgError
            If matrix is not invertible.
        """
        matrix = np.array(matrix)
        if not np.allclose(matrix, matrix.T, atol=1e-08):
            raise ValueError('Shape matrix must be symmetric.')
        np.linalg.inv(matrix)
        eigvals = np.linalg.eigvals(matrix)
        if np.min(eigvals) < 0:
            raise ValueError(f'Non positive-definite shape matrix (detected eigenvalues {eigvals})')
        for diag_entry in np.diagonal(matrix):
            if np.isnan(np.power(diag_entry, 0.5)):
                raise ValueError(f'Cannot evaluate square root of the diagonal entry {diag_entry} of argument `shape_matrix`. Check that this entry is nonnegative')

    @property
    def shape_matrix(self):
        """
        (N, N) numpy.ndarray : A positive definite matrix characterizing
        the shape and orientation of the ellipsoid.
        """
        return self._shape_matrix

    @shape_matrix.setter
    def shape_matrix(self, val):
        validate_array(arr=val, arr_name='shape_matrix', dim=2, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
        shape_mat_arr = np.array(val)
        if hasattr(self, '_center'):
            if not all((size == self.dim for size in shape_mat_arr.shape)):
                raise ValueError(f"EllipsoidalSet attribute 'shape_matrix' must be a square matrix of size {self.dim} to match set dimension (provided matrix with shape {shape_mat_arr.shape})")
        self._verify_positive_definite(shape_mat_arr)
        self._shape_matrix = shape_mat_arr

    @property
    def scale(self):
        """
        numeric type : Square of the factor by which to scale the
        semi-axes of the ellipsoid (i.e. the eigenvectors of the shape
        matrix).
        """
        return self._scale

    @scale.setter
    def scale(self, val):
        validate_arg_type('scale', val, valid_num_types, 'a valid numeric type', False)
        if val < 0:
            raise ValueError(f"EllipsoidalSet attribute 'scale' must be a non-negative real (provided value {val})")
        self._scale = val

    @property
    def dim(self):
        """
        int : Dimension `N` of the ellipsoidal set.
        """
        return len(self.center)

    @property
    def geometry(self):
        """
        Geometry of the ellipsoidal set.
        See the `Geometry` class documentation.
        """
        return Geometry.CONVEX_NONLINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the ellipsoidal set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        scale = self.scale
        nom_value = self.center
        P = self.shape_matrix
        parameter_bounds = [(nom_value[i] - np.power(P[i][i] * scale, 0.5), nom_value[i] + np.power(P[i][i] * scale, 0.5)) for i in range(self.dim)]
        return parameter_bounds

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of ellipsoidal constraints on a given sequence
        of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : {IndexedParam, IndexedVar, list of Param/Var}
            Uncertain parameter objects upon which the constraints
            are imposed. Indexed parameters are accepted, and
            are unpacked for constraint generation.
        **kwargs : dict, optional
            Additional arguments. These arguments are currently
            ignored.

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
        """
        inv_covar = np.linalg.inv(self.shape_matrix)
        if len(uncertain_params) != len(self.center):
            raise AttributeError('Center of ellipsoid must be same dimensions as vector of uncertain parameters.')
        diff = []
        for idx, i in enumerate(uncertain_params):
            if uncertain_params[idx].is_indexed():
                for index in uncertain_params[idx]:
                    diff.append(uncertain_params[idx][index] - self.center[idx])
            else:
                diff.append(uncertain_params[idx] - self.center[idx])
        product1 = [sum([x * y for x, y in zip(diff, column(inv_covar, i))]) for i in range(len(inv_covar))]
        constraint = sum([x * y for x, y in zip(product1, diff)])
        conlist = ConstraintList()
        conlist.construct()
        conlist.add(constraint <= self.scale)
        return conlist