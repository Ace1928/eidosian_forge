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
class IntersectionSet(UncertaintySet):
    """
    An intersection of a sequence of uncertainty sets, each of which
    is represented by an `UncertaintySet` object.

    Parameters
    ----------
    **unc_sets : dict
        PyROS `UncertaintySet` objects of which to construct
        an intersection. At least two uncertainty sets must
        be provided. All sets must be of the same dimension.

    Examples
    --------
    Intersection of origin-centered 2D box (square) and 2D
    hypersphere (circle):

    >>> from pyomo.contrib.pyros import (
    ...     BoxSet, AxisAlignedEllipsoidalSet, IntersectionSet,
    ... )
    >>> square = BoxSet(bounds=[[-1.5, 1.5], [-1.5, 1.5]])
    >>> circle = AxisAlignedEllipsoidalSet(
    ...     center=[0, 0],
    ...     half_lengths=[2, 2],
    ... )
    >>> # to construct intersection, pass sets as keyword arguments
    >>> intersection = IntersectionSet(set1=square, set2=circle)
    >>> intersection.all_sets
    UncertaintySetList([...])

    """

    def __init__(self, **unc_sets):
        """Initialize self (see class docstring)."""
        self.all_sets = unc_sets

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return 'intersection'

    @property
    def all_sets(self):
        """
        UncertaintySetList : List of the uncertainty sets of which to
        take the intersection. Must be of minimum length 2.

        This attribute may be set through any iterable of
        `UncertaintySet` objects, and exhibits similar behavior
        to a `list`.
        """
        return self._all_sets

    @all_sets.setter
    def all_sets(self, val):
        if isinstance(val, dict):
            the_sets = val.values()
        else:
            the_sets = list(val)
        all_sets = UncertaintySetList(the_sets, name='all_sets', min_length=2)
        if hasattr(self, '_all_sets'):
            if all_sets.dim != self.dim:
                raise ValueError(f"Attempting to set attribute 'all_sets' of an IntersectionSet of dimension {self.dim} to a sequence of sets of dimension {all_sets[0].dim}")
        self._all_sets = all_sets

    @property
    def dim(self):
        """
        int : Dimension of the intersection set.
        """
        return self.all_sets[0].dim

    @property
    def geometry(self):
        """
        Geometry of the intersection set.
        See the `Geometry` class documentation.
        """
        return max((self.all_sets[i].geometry.value for i in range(len(self.all_sets))))

    @property
    def parameter_bounds(self):
        """
        Uncertain parameter value bounds for the intersection
        set.

        Currently, an empty list, as the bounds cannot, in general,
        be computed without access to an optimization solver.
        """
        return []

    def point_in_set(self, point):
        """
        Determine whether a given point lies in the intersection set.

        Parameters
        ----------
        point : (N,) array-like
            Point (parameter value) of interest.

        Returns
        -------
        : bool
            True if the point lies in the set, False otherwise.
        """
        if all((a_set.point_in_set(point=point) for a_set in self.all_sets)):
            return True
        else:
            return False

    def is_empty_intersection(self, uncertain_params, nlp_solver):
        """
        Determine if intersection is empty.

        Arguments
        ---------
        uncertain_params : list of Param or list of Var
            List of uncertain parameter objects.
        nlp_solver : Pyomo SolverFactory object
            NLP solver.

        Returns
        -------
        is_empty_intersection : bool
            True if the intersection is certified to be empty,
            and False otherwise.
        """
        is_empty_intersection = True
        if any((a_set.type == 'discrete' for a_set in self.all_sets)):
            disc_sets = (a_set for a_set in self.all_sets if a_set.type == 'discrete')
            disc_set = min(disc_sets, key=lambda x: len(x.scenarios))
            for scenario in disc_set.scenarios:
                if all((a_set.point_in_set(point=scenario) for a_set in self.all_sets)):
                    is_empty_intersection = False
                    break
        else:
            m = ConcreteModel()
            m.obj = Objective(expr=0)
            m.param_vars = Var(uncertain_params.index_set())
            for a_set in self.all_sets:
                m.add_component(a_set.type + '_constraints', a_set.set_as_constraint(uncertain_params=m.param_vars))
            try:
                res = nlp_solver.solve(m)
            except:
                raise ValueError('Solver terminated with an error while checking set intersection non-emptiness.')
            if check_optimal_termination(res):
                is_empty_intersection = False
        return is_empty_intersection

    @staticmethod
    def intersect(Q1, Q2):
        """
        Obtain the intersection of two uncertainty sets.

        Parameters
        ----------
        Q1, Q2 : UncertaintySet
            Operand uncertainty sets.

        Returns
        -------
        : DiscreteScenarioSet or IntersectionSet
            Intersection of the sets. A `DiscreteScenarioSet` is
            returned if both operand sets are `DiscreteScenarioSet`
            instances; otherwise, an `IntersectionSet` is returned.
        """
        constraints = ConstraintList()
        constraints.construct()
        for set in (Q1, Q2):
            other = Q1 if set is Q2 else Q2
            if set.type == 'discrete':
                intersected_scenarios = []
                for point in set.scenarios:
                    if other.point_in_set(point=point):
                        intersected_scenarios.append(point)
                return DiscreteScenarioSet(scenarios=intersected_scenarios)
        return IntersectionSet(set1=Q1, set2=Q2)
        return

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of constraints on a given sequence
        of uncertain parameter objects. In advance of constructing
        the constraints, a check is performed to determine whether
        the set is empty.

        Parameters
        ----------
        uncertain_params : list of Param or list of Var
            Uncertain parameter objects upon which the constraints
            are imposed.
        **kwargs : dict
            Additional arguments. Must contain a `config` entry,
            which maps to a `ConfigDict` containing an entry
            entitled `global_solver`. The `global_solver`
            key maps to an NLP solver, purportedly with global
            optimization capabilities.

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.

        Raises
        ------
        AttributeError
            If the intersection set is found to be empty.
        """
        try:
            nlp_solver = kwargs['config'].global_solver
        except:
            raise AttributeError('set_as_constraint for SetIntersection requires access to an NLP solver viathe PyROS Solver config.')
        is_empty_intersection = self.is_empty_intersection(uncertain_params=uncertain_params, nlp_solver=nlp_solver)

        def _intersect(Q1, Q2):
            return self.intersect(Q1, Q2)
        if not is_empty_intersection:
            Qint = functools.reduce(_intersect, self.all_sets)
            if Qint.type == 'discrete':
                return Qint.set_as_constraint(uncertain_params=uncertain_params)
            else:
                conlist = ConstraintList()
                conlist.construct()
                for set in Qint.all_sets:
                    for con in list(set.set_as_constraint(uncertain_params=uncertain_params).values()):
                        conlist.add(con.expr)
                return conlist
        else:
            raise AttributeError('Set intersection is empty, cannot proceed with PyROS robust optimization.')

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
        return