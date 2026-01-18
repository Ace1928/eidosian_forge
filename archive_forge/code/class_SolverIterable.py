from collections.abc import Iterable
import logging
from pyomo.common.collections import ComponentSet
from pyomo.common.config import (
from pyomo.common.errors import ApplicationError, PyomoException
from pyomo.core.base import Var, _VarData
from pyomo.core.base.param import Param, _ParamData
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.util import ObjectiveType, setup_pyros_logger
from pyomo.contrib.pyros.uncertainty_sets import UncertaintySet
class SolverIterable(object):
    """
    Callable for casting an iterable (such as a list of strs)
    to a list of Pyomo solvers.

    Parameters
    ----------
    require_available : bool, optional
        True if `available()` method of a standardized solver
        object obtained through `self` must return `True`,
        False otherwise.
    filter_by_availability : bool, optional
        True to remove standardized solvers for which `available()`
        does not return True, False otherwise.
    solver_desc : str, optional
        Descriptor for the solver obtained through `self`,
        such as 'backup local solver'
        or 'backup global solver'.
    """

    def __init__(self, require_available=True, filter_by_availability=True, solver_desc='solver'):
        """Initialize self (see class docstring)."""
        self.require_available = require_available
        self.filter_by_availability = filter_by_availability
        self.solver_desc = solver_desc

    def __call__(self, obj, require_available=None, filter_by_availability=None, solver_desc=None):
        """
        Cast iterable object to a list of Pyomo solver objects.

        Parameters
        ----------
        obj : str, Solver, or Iterable of str/Solver
            Object of interest.
        require_available : bool or None, optional
            True if `available()` method of each solver
            object must return True, False otherwise.
            If `None` is passed, then ``self.require_available``
            is used.
        solver_desc : str or None, optional
            Descriptor for the solver, such as 'backup local solver'
            or 'backup global solver'. This argument is used
            for constructing error/exception messages.
            If `None` is passed, then ``self.solver_desc``
            is used.

        Returns
        -------
        solvers : list of solver type
            List of solver objects to which obj is cast.

        Raises
        ------
        TypeError
            If `obj` is a str.
        """
        if require_available is None:
            require_available = self.require_available
        if filter_by_availability is None:
            filter_by_availability = self.filter_by_availability
        if solver_desc is None:
            solver_desc = self.solver_desc
        solver_resolve_func = SolverResolvable()
        if isinstance(obj, str) or solver_resolve_func.is_solver_type(obj):
            obj_as_list = [obj]
        else:
            obj_as_list = list(obj)
        solvers = []
        for idx, val in enumerate(obj_as_list):
            solver_desc_str = f'{solver_desc} (index {idx})'
            opt = solver_resolve_func(obj=val, require_available=require_available, solver_desc=solver_desc_str)
            if filter_by_availability and (not opt.available(exception_flag=False)):
                default_pyros_solver_logger.warning(f'Output of `available()` method for solver object {opt} resolved from object {val} of sequence {obj_as_list} to be used as {self.solver_desc} is not `True`. Removing from list of standardized solvers.')
            else:
                solvers.append(opt)
        return solvers

    def domain_name(self):
        """Return str briefly describing domain encompassed by self."""
        return 'str, solver type, or Iterable of str/solver type'