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
class AxisAlignedEllipsoidalSet(UncertaintySet):
    """
    An axis-aligned ellipsoid.

    Parameters
    ----------
    center : (N,) array_like
        Center of the ellipsoid.
    half_lengths : (N,) array_like
        Semi-axis lengths of the ellipsoid.

    Examples
    --------
    3D origin-centered unit hypersphere:

    >>> from pyomo.contrib.pyros import AxisAlignedEllipsoidalSet
    >>> sphere = AxisAlignedEllipsoidalSet(
    ...     center=[0, 0, 0],
    ...     half_lengths=[1, 1, 1]
    ... )
    >>> sphere.center
    array([0, 0, 0])
    >>> sphere.half_lengths
    array([1, 1, 1])

    """

    def __init__(self, center, half_lengths):
        """Initialize self (see class docstring)."""
        self.center = center
        self.half_lengths = half_lengths

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

    @property
    def half_lengths(self):
        """
        (N,) numpy.ndarray : Semi-axis lengths.
        """
        return self._half_lengths

    @half_lengths.setter
    def half_lengths(self, val):
        validate_array(arr=val, arr_name='half_lengths', dim=1, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
        val_arr = np.array(val)
        if hasattr(self, '_center'):
            if val_arr.size != self.dim:
                raise ValueError(f"Attempting to set attribute 'half_lengths' of AxisAlignedEllipsoidalSet of dimension {self.dim} to value of dimension {val_arr.size}")
        for half_len in val_arr:
            if half_len < 0:
                raise ValueError(f"Entry {half_len} of 'half_lengths' is negative. All half-lengths must be nonnegative")
        self._half_lengths = val_arr

    @property
    def dim(self):
        """
        int : Dimension `N` of the axis-aligned ellipsoidal set.
        """
        return len(self.center)

    @property
    def geometry(self):
        """
        Geometry of the axis-aligned ellipsoidal set.
        See the `Geometry` class documentation.
        """
        return Geometry.CONVEX_NONLINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the axis-aligned ellipsoidal set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        nom_value = self.center
        half_length = self.half_lengths
        parameter_bounds = [(nom_value[i] - half_length[i], nom_value[i] + half_length[i]) for i in range(len(nom_value))]
        return parameter_bounds

    def set_as_constraint(self, uncertain_params, model=None, config=None):
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
        all_params = list()
        if not isinstance(uncertain_params, (tuple, list)):
            uncertain_params = [uncertain_params]
        all_params = []
        for uparam in uncertain_params:
            all_params.extend(uparam.values())
        if len(all_params) != len(self.center):
            raise AttributeError(f'Center of ellipsoid is of dimension {len(self.center)}, but vector of uncertain parameters is of dimension {len(all_params)}')
        zip_all = zip(all_params, self.center, self.half_lengths)
        diffs_squared = list()
        conlist = ConstraintList()
        conlist.construct()
        for param, ctr, half_len in zip_all:
            if half_len > 0:
                diffs_squared.append((param - ctr) ** 2 / half_len ** 2)
            else:
                conlist.add(param == ctr)
        conlist.add(sum(diffs_squared) <= 1)
        return conlist