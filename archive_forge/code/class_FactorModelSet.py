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
class FactorModelSet(UncertaintySet):
    """
    A factor model (a.k.a. "net-alpha" model) set.

    Parameters
    ----------
    origin : (N,) array_like
        Uncertain parameter values around which deviations are
        restrained.
    number_of_factors : int
        Natural number representing the dimensionality of the
        space to which the set projects.
    psi_mat : (N, F) array_like
        Matrix designating each uncertain parameter's contribution to
        each factor.  Each row is associated with a separate uncertain
        parameter.  Each column is associated with a separate factor.
        Number of columns `F` of `psi_mat` should be equal to
        `number_of_factors`.
    beta : numeric type
        Real value between 0 and 1 specifying the fraction of the
        independent factors that can simultaneously attain
        their extreme values.

    Examples
    --------
    A 4D factor model set with a 2D factor space:

    >>> from pyomo.contrib.pyros import FactorModelSet
    >>> import numpy as np
    >>> fset = FactorModelSet(
    ...     origin=np.zeros(4),
    ...     number_of_factors=2,
    ...     psi_mat=np.full(shape=(4, 2), fill_value=0.1),
    ...     beta=0.5,
    ... )
    >>> fset.origin
    array([0., 0., 0., 0.])
    >>> fset.number_of_factors
    2
    >>> fset.psi_mat
    array([[0.1, 0.1],
           [0.1, 0.1],
           [0.1, 0.1],
           [0.1, 0.1]])
    >>> fset.beta
    0.5
    """

    def __init__(self, origin, number_of_factors, psi_mat, beta):
        """Initialize self (see class docstring)."""
        self.origin = origin
        self.number_of_factors = number_of_factors
        self.beta = beta
        self.psi_mat = psi_mat

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return 'factor_model'

    @property
    def origin(self):
        """
        (N,) numpy.ndarray : Uncertain parameter values around which
        deviations are restrained.
        """
        return self._origin

    @origin.setter
    def origin(self, val):
        validate_array(arr=val, arr_name='origin', dim=1, valid_types=valid_num_types, valid_type_desc='a valid numeric type')
        val_arr = np.array(val)
        if hasattr(self, '_origin'):
            if val_arr.size != self.dim:
                raise ValueError(f"Attempting to set attribute 'origin' of factor model set of dimension {self.dim} to value of dimension {val_arr.size}")
        self._origin = val_arr

    @property
    def number_of_factors(self):
        """
        int : Natural number representing the dimensionality `F`
        of the space to which the set projects.

        This attribute is immutable, and may only be set at
        object construction. Typically, the number of factors
        is significantly less than the set dimension, but no
        restriction to that end is imposed here.
        """
        return self._number_of_factors

    @number_of_factors.setter
    def number_of_factors(self, val):
        if hasattr(self, '_number_of_factors'):
            raise AttributeError("Attribute 'number_of_factors' is immutable")
        else:
            validate_arg_type('number_of_factors', val, Integral)
            if val < 1:
                raise ValueError(f"Attribute 'number_of_factors' must be a positive int (provided value {val})")
        self._number_of_factors = val

    @property
    def psi_mat(self):
        """
        (N, F) numpy.ndarray : Matrix designating each
        uncertain parameter's contribution to each factor. Each row is
        associated with a separate uncertain parameter. Each column with
        a separate factor.
        """
        return self._psi_mat

    @psi_mat.setter
    def psi_mat(self, val):
        validate_array(arr=val, arr_name='psi_mat', dim=2, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
        psi_mat_arr = np.array(val)
        if psi_mat_arr.shape != (self.dim, self.number_of_factors):
            raise ValueError(f'Psi matrix for factor model set should be of shape {(self.dim, self.number_of_factors)} to match the set and factor model space dimensions (provided shape {psi_mat_arr.shape})')
        for column in psi_mat_arr.T:
            if np.allclose(column, 0):
                raise ValueError("Each column of attribute 'psi_mat' should have at least one nonzero entry")
        self._psi_mat = psi_mat_arr

    @property
    def beta(self):
        """
        numeric type : Real number ranging from 0 to 1 representing the
        fraction of the independent factors that can simultaneously
        attain their extreme values.

        Note that, mathematically, setting ``beta = 0`` will enforce
        that as many factors will be above 0 as there will be below 0
        (i.e., "zero-net-alpha" model). If ``beta = 1``,
        then the set is numerically equivalent to a `BoxSet` with bounds
        ``[origin - psi @ np.ones(F), origin + psi @ np.ones(F)].T``.
        """
        return self._beta

    @beta.setter
    def beta(self, val):
        if val > 1 or val < 0:
            raise ValueError(f'Beta parameter must be a real number between 0 and 1 inclusive (provided value {val})')
        self._beta = val

    @property
    def dim(self):
        """
        int : Dimension `N` of the factor model set.
        """
        return len(self.origin)

    @property
    def geometry(self):
        """
        Geometry of the factor model set.
        See the `Geometry` class documentation.
        """
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the factor model set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        F = self.number_of_factors
        psi_mat = self.psi_mat
        beta_F = self.beta * self.number_of_factors
        crit_pt_type = int((beta_F + F) / 2)
        beta_F_fill_in = beta_F + F - 2 * crit_pt_type - 1
        row_wise_args = np.argsort(-psi_mat, axis=1)
        parameter_bounds = []
        for idx, orig_val in enumerate(self.origin):
            M = len(psi_mat[idx][psi_mat[idx] >= 0])
            sorted_psi_row_args = row_wise_args[idx]
            sorted_psi_row = psi_mat[idx, sorted_psi_row_args]
            if M > crit_pt_type:
                max_deviation = sorted_psi_row[:crit_pt_type].sum() + beta_F_fill_in * sorted_psi_row[crit_pt_type] - sorted_psi_row[crit_pt_type + 1:].sum()
            elif M < F - crit_pt_type:
                max_deviation = sorted_psi_row[:F - crit_pt_type - 1].sum() - beta_F_fill_in * sorted_psi_row[F - crit_pt_type - 1] - sorted_psi_row[F - crit_pt_type:].sum()
            else:
                max_deviation = sorted_psi_row[:M].sum() - sorted_psi_row[M:].sum()
            parameter_bounds.append((orig_val - max_deviation, orig_val + max_deviation))
        return parameter_bounds

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of factor model constraints on a given sequence
        of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : list of Param or list of Var
            Uncertain parameter objects upon which the constraints
            are imposed.
        **kwargs : dict
            Additional arguments. This dictionary should consist
            of a `model` entry, which maps to a `ConcreteModel`
            object representing the model of interest (parent model
            of the uncertain parameter objects).

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
        """
        model = kwargs['model']
        if len(uncertain_params) != len(self.origin):
            raise AttributeError('Dimensions of origin and uncertain_param lists must be equal.')
        n = list(range(self.number_of_factors))
        model.util.cassi = Var(n, initialize=0, bounds=(-1, 1))
        conlist = ConstraintList()
        conlist.construct()
        disturbances = [sum((self.psi_mat[i][j] * model.util.cassi[j] for j in n)) for i in range(len(uncertain_params))]
        for i in range(len(uncertain_params)):
            conlist.add(self.origin[i] + disturbances[i] == uncertain_params[i])
        conlist.add(sum((model.util.cassi[i] for i in n)) <= +self.beta * self.number_of_factors)
        conlist.add(sum((model.util.cassi[i] for i in n)) >= -self.beta * self.number_of_factors)
        return conlist

    def point_in_set(self, point):
        """
        Determine whether a given point lies in the factor model set.

        Parameters
        ----------
        point : (N,) array-like
            Point (parameter value) of interest.

        Returns
        -------
        : bool
            True if the point lies in the set, False otherwise.
        """
        inv_psi = np.linalg.pinv(self.psi_mat)
        diff = np.asarray(list((point[i] - self.origin[i] for i in range(len(point)))))
        cassis = np.dot(inv_psi, np.transpose(diff))
        if abs(sum((cassi for cassi in cassis))) <= self.beta * self.number_of_factors and all((cassi >= -1 and cassi <= 1 for cassi in cassis)):
            return True
        else:
            return False