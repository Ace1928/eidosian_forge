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
class BoxSet(UncertaintySet):
    """
    A hyper-rectangle (a.k.a. "box").

    Parameters
    ----------
    bounds : (N, 2) array_like
        Lower and upper bounds for each dimension of the set.

    Examples
    --------
    1D box set (interval):

    >>> from pyomo.contrib.pyros import BoxSet
    >>> interval = BoxSet(bounds=[(1, 2)])
    >>> interval.bounds
    array([[1, 2]])

    2D box set:

    >>> box_set = BoxSet(bounds=[[1, 2], [3, 4]])
    >>> box_set.bounds
    array([[1, 2],
           [3, 4]])

    5D hypercube with bounds 0 and 1 in each dimension:

    >>> hypercube_5d = BoxSet(bounds=[[0, 1] for idx in range(5)])
    >>> hypercube_5d.bounds
    array([[0, 1],
           [0, 1],
           [0, 1],
           [0, 1],
           [0, 1]])
    """

    def __init__(self, bounds):
        """Initialize self (see class docstring)."""
        self.bounds = bounds

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return 'box'

    @property
    def bounds(self):
        """
        (N, 2) numpy.ndarray : Lower and upper bounds for each dimension
        of the set.

        The bounds of a `BoxSet` instance can be changed, such that
        the dimension of the set remains unchanged.
        """
        return self._bounds

    @bounds.setter
    def bounds(self, val):
        validate_array(arr=val, arr_name='bounds', dim=2, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=[None, 2])
        bounds_arr = np.array(val)
        for lb, ub in bounds_arr:
            if lb > ub:
                raise ValueError(f'Lower bound {lb} exceeds upper bound {ub}')
        if hasattr(self, '_bounds') and bounds_arr.shape[0] != self.dim:
            raise ValueError(f'Attempting to set bounds of a box set of dimension {self.dim} to a value of dimension {bounds_arr.shape[0]}')
        self._bounds = np.array(val)

    @property
    def dim(self):
        """
        int : Dimension `N` of the box set.
        """
        return len(self.bounds)

    @property
    def geometry(self):
        """
        Geometry of the box set.
        See the `Geometry` class documentation.
        """
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the box set.
        This is numerically equivalent to the `bounds` attribute.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        return [tuple(bound) for bound in self.bounds]

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of box constraints on a given sequence
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
        conlist = ConstraintList()
        conlist.construct()
        set_i = list(range(len(uncertain_params)))
        for i in set_i:
            conlist.add(uncertain_params[i] >= self.bounds[i][0])
            conlist.add(uncertain_params[i] <= self.bounds[i][1])
        return conlist