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
class DiscreteScenarioSet(UncertaintySet):
    """
    A discrete set of finitely many uncertain parameter realizations
    (or scenarios).

    Parameters
    ----------
    scenarios : (M, N) array_like
        A sequence of `M` distinct uncertain parameter realizations.

    Examples
    --------
    2D set with three scenarios:

    >>> from pyomo.contrib.pyros import DiscreteScenarioSet
    >>> discrete_set = DiscreteScenarioSet(
    ...     scenarios=[[1, 1], [2, 1], [1, 2]],
    ... )
    >>> discrete_set.scenarios
    [(1, 1), (2, 1), (1, 2)]

    """

    def __init__(self, scenarios):
        """Initialize self (see class docstring)."""
        self.scenarios = scenarios

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return 'discrete'

    @property
    def scenarios(self):
        """
        list of tuples : Uncertain parameter realizations comprising the
        set.  Each tuple is an uncertain parameter realization.

        Note that the `scenarios` attribute may be modified, but
        only such that the dimension of the set remains unchanged.
        """
        return self._scenarios

    @scenarios.setter
    def scenarios(self, val):
        validate_array(arr=val, arr_name='scenarios', dim=2, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
        scenario_arr = np.array(val)
        if hasattr(self, '_scenarios'):
            if scenario_arr.shape[1] != self.dim:
                raise ValueError(f"DiscreteScenarioSet attribute 'scenarios' must have {self.dim} columns to match set dimension (provided array-like with {scenario_arr.shape[1]} columns)")
        self._scenarios = [tuple(s) for s in val]

    @property
    def dim(self):
        """
        int : Dimension `N` of the discrete scenario set.
        """
        return len(self.scenarios[0])

    @property
    def geometry(self):
        """
        Geometry of the discrete scenario set.
        See the `Geometry` class documentation.
        """
        return Geometry.DISCRETE_SCENARIOS

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the discrete scenario set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        parameter_bounds = [(min((s[i] for s in self.scenarios)), max((s[i] for s in self.scenarios))) for i in range(self.dim)]
        return parameter_bounds

    def is_bounded(self, config):
        """
        Return True if the uncertainty set is bounded, and False
        otherwise.

        By default, the discrete scenario set is bounded,
        as the entries of all uncertain parameter scenarios
        are finite.
        """
        return True

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of constraints on a given sequence
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
        dim = len(uncertain_params)
        if any((len(d) != dim for d in self.scenarios)):
            raise AttributeError('All scenarios must have same dimensions as uncertain parameters.')
        conlist = ConstraintList()
        conlist.construct()
        for n in list(range(len(self.scenarios))):
            for i in list(range(len(uncertain_params))):
                conlist.add(uncertain_params[i] == self.scenarios[n][i])
        conlist.deactivate()
        return conlist

    def point_in_set(self, point):
        """
        Determine whether a given point lies in the discrete
        scenario set.

        Parameters
        ----------
        point : (N,) array-like
            Point (parameter value) of interest.

        Returns
        -------
        : bool
            True if the point lies in the set, False otherwise.
        """
        num_decimals = 8
        rounded_scenarios = list((list((round(num, num_decimals) for num in d)) for d in self.scenarios))
        rounded_point = list((round(num, num_decimals) for num in point))
        return any((rounded_point == rounded_d for rounded_d in rounded_scenarios))