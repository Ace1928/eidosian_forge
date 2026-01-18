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
class BudgetSet(UncertaintySet):
    """
    A budget set.

    Parameters
    ----------
    budget_membership_mat : (L, N) array_like
        Incidence matrix of the budget constraints.
        Each row corresponds to a single budget constraint,
        and defines which uncertain parameters
        (which dimensions) participate in that row's constraint.
    rhs_vec : (L,) array_like
        Budget limits (upper bounds) with respect to
        the origin of the set.
    origin : (N,) array_like or None, optional
        Origin of the budget set. If `None` is provided, then
        the origin is set to the zero vector.

    Examples
    --------
    3D budget set with one budget constraint and
    no origin chosen (hence origin defaults to 3D zero vector):

    >>> from pyomo.contrib.pyros import BudgetSet
    >>> budget_set = BudgetSet(
    ...     budget_membership_mat=[[1, 1, 1]],
    ...     rhs_vec=[2],
    ... )
    >>> budget_set.budget_membership_mat
    array([[1, 1, 1]])
    >>> budget_set.budget_rhs_vec
    array([2])
    >>> budget_set.origin
    array([0., 0., 0.])

    3D budget set with two budget constraints and custom origin:

    >>> budget_custom = BudgetSet(
    ...     budget_membership_mat=[[1, 0, 1], [0, 1, 0]],
    ...     rhs_vec=[1, 1],
    ...     origin=[2, 2, 2],
    ... )
    >>> budget_custom.budget_membership_mat
    array([[1, 0, 1],
           [0, 1, 0]])
    >>> budget_custom.budget_rhs_vec
    array([1, 1])
    >>> budget_custom.origin
    array([2, 2, 2])
    """

    def __init__(self, budget_membership_mat, rhs_vec, origin=None):
        """Initialize self (see class docstring)."""
        self.budget_membership_mat = budget_membership_mat
        self.budget_rhs_vec = rhs_vec
        self.origin = np.zeros(self.dim) if origin is None else origin

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return 'budget'

    @property
    def coefficients_mat(self):
        """
        (L + N, N) numpy.ndarray : Coefficient matrix of all polyhedral
        constraints defining the budget set. Composed from the incidence
        matrix used for defining the budget constraints and a
        coefficient matrix for individual uncertain parameter
        nonnegativity constraints.

        This attribute cannot be set. The budget constraint
        incidence matrix may be altered through the
        `budget_membership_mat` attribute.
        """
        return np.append(self.budget_membership_mat, -np.identity(self.dim), axis=0)

    @property
    def rhs_vec(self):
        """
        (L + N,) numpy.ndarray : Right-hand side vector for polyhedral
        constraints defining the budget set. This also includes entries
        for nonnegativity constraints on the uncertain parameters.

        This attribute cannot be set, and is automatically determined
        given other attributes.
        """
        return np.append(self.budget_rhs_vec + self.budget_membership_mat @ self.origin, -self.origin)

    @property
    def budget_membership_mat(self):
        """
        (L, N) numpy.ndarray : Incidence matrix of the budget
        constraints.  Each row corresponds to a single budget
        constraint and defines which uncertain parameters
        participate in that row's constraint.
        """
        return self._budget_membership_mat

    @budget_membership_mat.setter
    def budget_membership_mat(self, val):
        validate_array(arr=val, arr_name='budget_membership_mat', dim=2, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
        lhs_coeffs_arr = np.array(val)
        if hasattr(self, '_budget_membership_mat'):
            if lhs_coeffs_arr.shape[1] != self.dim:
                raise ValueError(f"BudgetSet attribute 'budget_membership_mat' must have {self.dim} columns to match set dimension (provided matrix with {lhs_coeffs_arr.shape[1]} columns)")
        if hasattr(self, '_budget_rhs_vec'):
            if lhs_coeffs_arr.shape[0] != self.budget_rhs_vec.size:
                raise ValueError(f"BudgetSet attribute 'budget_membership_mat' must have {self.budget_rhs_vec.size} rows to match shape of attribute 'budget_rhs_vec' (provided {lhs_coeffs_arr.shape[0]} rows)")
        uniq_entries = np.unique(lhs_coeffs_arr)
        non_bool_entries = uniq_entries[(uniq_entries != 0) & (uniq_entries != 1)]
        if non_bool_entries.size > 0:
            raise ValueError(f'Attempting to set attribute `budget_membership_mat` to value containing entries that are not 0-1 values (example: {non_bool_entries[0]}). Ensure all entries are of value 0 or 1')
        rows_with_zero_sums = np.nonzero(lhs_coeffs_arr.sum(axis=1) == 0)[0]
        if rows_with_zero_sums.size > 0:
            row_str = ', '.join((str(val) for val in rows_with_zero_sums))
            raise ValueError(f'Attempting to set attribute `budget_membership_mat` to value with all entries zero in rows at indexes: {row_str}. Ensure each row and column has at least one nonzero entry')
        cols_with_zero_sums = np.nonzero(lhs_coeffs_arr.sum(axis=0) == 0)[0]
        if cols_with_zero_sums.size > 0:
            col_str = ', '.join((str(val) for val in cols_with_zero_sums))
            raise ValueError(f'Attempting to set attribute `budget_membership_mat` to value with all entries zero in columns at indexes: {col_str}. Ensure each row and column has at least one nonzero entry')
        self._budget_membership_mat = lhs_coeffs_arr

    @property
    def budget_rhs_vec(self):
        """
        (L,) numpy.ndarray : Budget limits (upper bounds)
        with respect to the origin.
        """
        return self._budget_rhs_vec

    @budget_rhs_vec.setter
    def budget_rhs_vec(self, val):
        validate_array(arr=val, arr_name='budget_rhs_vec', dim=1, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
        rhs_vec_arr = np.array(val)
        if hasattr(self, '_budget_membership_mat'):
            if len(val) != self.budget_membership_mat.shape[0]:
                raise ValueError(f"Budget set attribute 'budget_rhs_vec' must have {self.budget_membership_mat.shape[0]} entries to match shape of attribute 'budget_membership_mat' (provided {rhs_vec_arr.size} entries)")
        for entry in rhs_vec_arr:
            if entry < 0:
                raise ValueError(f"Entry {entry} of attribute 'budget_rhs_vec' is negative. Ensure all entries are nonnegative")
        self._budget_rhs_vec = rhs_vec_arr

    @property
    def origin(self):
        """
        (N,) numpy.ndarray : Origin of the budget set.
        """
        return self._origin

    @origin.setter
    def origin(self, val):
        validate_array(arr=val, arr_name='origin', dim=1, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
        origin_arr = np.array(val)
        if len(val) != self.dim:
            raise ValueError(f"Budget set attribute 'origin' must have {self.dim} entries to match set dimension (provided {origin_arr.size} entries)")
        self._origin = origin_arr

    @property
    def dim(self):
        """
        int : Dimension `N` of the budget set.
        """
        return self.budget_membership_mat.shape[1]

    @property
    def geometry(self):
        """
        Geometry of the budget set.
        See the `Geometry` class documentation.
        """
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the budget set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        bounds = []
        for orig_val, col in zip(self.origin, self.budget_membership_mat.T):
            lb = orig_val
            ub = orig_val + np.min(self.budget_rhs_vec[col == 1])
            bounds.append((lb, ub))
        return bounds

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of the constraints defining the budget
        set on a given sequence of uncertain parameter objects.

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
        if self.dim != len(uncertain_params):
            raise ValueError(f"Argument 'uncertain_params' must contain {self.dim}Param objects to match BudgetSet dimension(provided {len(uncertain_params)} objects)")
        return PolyhedralSet.set_as_constraint(self, uncertain_params)

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
        UncertaintySet.add_bounds_on_uncertain_parameters(model=model, config=config)