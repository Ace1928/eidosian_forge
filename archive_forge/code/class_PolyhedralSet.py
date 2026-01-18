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
class PolyhedralSet(UncertaintySet):
    """
    A bounded convex polyhedron or polytope.

    Parameters
    ----------
    lhs_coefficients_mat : (M, N) array_like
        Left-hand side coefficients for the linear
        inequality constraints defining the polyhedral set.
    rhs_vec : (M,) array_like
        Right-hand side values for the linear inequality
        constraints defining the polyhedral set.
        Each entry is an upper bound for the quantity
        ``lhs_coefficients_mat @ x``, where `x` is an (N,)
        array representing any point in the polyhedral set.

    Examples
    --------
    2D polyhedral set with 4 defining inequalities:

    >>> from pyomo.contrib.pyros import PolyhedralSet
    >>> pset = PolyhedralSet(
    ...     lhs_coefficients_mat=[[-1, 0], [0, -1], [-1, 1], [1, 0]],
    ...     rhs_vec=[0, 0, 0, 1],
    ... )
    >>> pset.coefficients_mat
    array([[-1,  0],
           [ 0, -1],
           [-1,  1],
           [ 1,  0]])
    >>> pset.rhs_vec
    array([0, 0, 0, 1])
    """

    def __init__(self, lhs_coefficients_mat, rhs_vec):
        """Initialize self (see class docstring)."""
        self.coefficients_mat = lhs_coefficients_mat
        self.rhs_vec = rhs_vec
        self._validate()

    def _validate(self):
        """
        Check polyhedral set attributes are such that set is nonempty
        (solve a feasibility problem).

        Raises
        ------
        ValueError
            If set is empty, or the check was not
            successfully completed due to numerical issues.
        """
        res = sp.optimize.linprog(c=np.zeros(self.coefficients_mat.shape[1]), A_ub=self.coefficients_mat, b_ub=self.rhs_vec, method='simplex', bounds=(None, None))
        if res.status == 1 or res.status == 4:
            raise ValueError(f'Could not verify nonemptiness of the polyhedral set (`scipy.optimize.linprog(method=simplex)`  status {res.status}) ')
        elif res.status == 2:
            raise ValueError("PolyhedralSet defined by 'coefficients_mat' and 'rhs_vec' is empty. Check arguments")

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return 'polyhedral'

    @property
    def coefficients_mat(self):
        """
        (M, N) numpy.ndarray : Coefficient matrix for the (linear)
        inequality constraints defining the polyhedral set.

        In tandem with the `rhs_vec` attribute, this matrix should
        be such that the polyhedral set is nonempty and bounded.
        Such a check is performed only at instance construction.
        """
        return self._coefficients_mat

    @coefficients_mat.setter
    def coefficients_mat(self, val):
        validate_array(arr=val, arr_name='coefficients_mat', dim=2, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
        lhs_coeffs_arr = np.array(val)
        if hasattr(self, '_coefficients_mat'):
            if lhs_coeffs_arr.shape[1] != self.dim:
                raise ValueError(f"Polyhedral set attribute 'coefficients_mat' must have {self.dim} columns to match set dimension (provided matrix with {lhs_coeffs_arr.shape[1]} columns)")
        if hasattr(self, '_rhs_vec'):
            if lhs_coeffs_arr.shape[0] != self.rhs_vec.size:
                raise ValueError(f"PolyhedralSet attribute 'coefficients_mat' must have {self.rhs_vec.size} rows to match shape of attribute 'rhs_vec' (provided {lhs_coeffs_arr.shape[0]} rows)")
        cols_with_all_zeros = np.nonzero([np.all(col == 0) for col in lhs_coeffs_arr.T])[0]
        if cols_with_all_zeros.size > 0:
            col_str = ', '.join((str(val) for val in cols_with_all_zeros))
            raise ValueError(f"Attempting to set attribute 'coefficients_mat' to value with all entries zero in columns at indexes: {col_str}. Ensure column has at least one nonzero entry")
        self._coefficients_mat = lhs_coeffs_arr

    @property
    def rhs_vec(self):
        """
        (M,) numpy.ndarray : Right-hand side values (upper bounds) for
        the (linear) inequality constraints defining the polyhedral set.
        """
        return self._rhs_vec

    @rhs_vec.setter
    def rhs_vec(self, val):
        validate_array(arr=val, arr_name='rhs_vec', dim=1, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
        rhs_vec_arr = np.array(val)
        if hasattr(self, '_coefficients_mat'):
            if len(val) != self.coefficients_mat.shape[0]:
                raise ValueError(f"PolyhedralSet attribute 'rhs_vec' must have {self.coefficients_mat.shape[0]} entries to match shape of attribute 'coefficients_mat' (provided {rhs_vec_arr.size} entries)")
        self._rhs_vec = rhs_vec_arr

    @property
    def dim(self):
        """
        int : Dimension `N` of the polyhedral set.
        """
        return len(self.coefficients_mat[0])

    @property
    def geometry(self):
        """
        Geometry of the polyhedral set.
        See the `Geometry` class documentation.
        """
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the polyhedral set.

        Currently, an empty `list` is returned, as the bounds cannot, in
        general, be computed without access to an optimization solver.
        """
        return []

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of polyhedral constraints on a given sequence
        of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : list of Param or list of Var
            Uncertain parameter objects upon which the constraints
            are imposed.
        **kwargs : dict, optional
            Additional arguments. These arguments are currently
            ignored.

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
        """
        if np.asarray(self.coefficients_mat).shape[1] != len(uncertain_params):
            raise AttributeError('Columns of coefficients_mat matrix must equal length of uncertain parameters list.')
        set_i = list(range(len(self.coefficients_mat)))
        conlist = ConstraintList()
        conlist.construct()
        for i in set_i:
            constraint = 0
            for j in range(len(uncertain_params)):
                constraint += float(self.coefficients_mat[i][j]) * uncertain_params[j]
            conlist.add(constraint <= float(self.rhs_vec[i]))
        return conlist

    @staticmethod
    def add_bounds_on_uncertain_parameters(model, config):
        """
        Specify the numerical bounds for each of a sequence of uncertain
        parameters, represented by Pyomo `Var` objects, in a modeling
        object. The numerical bounds are specified through the `.lb()`
        and `.ub()` attributes of the `Var` objects.

        Parameters
        ----------
        model : ConcreteModel
            Model of interest (parent model of the uncertain parameter
            objects for which to specify bounds).
        config : ConfigDict
            PyROS solver config.

        Notes
        -----
        This method is invoked in advance of a PyROS separation
        subproblem.
        """
        add_bounds_for_uncertain_parameters(model=model, config=config)