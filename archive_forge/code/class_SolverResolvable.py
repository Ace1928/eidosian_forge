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
class SolverResolvable(object):
    """
    Callable for casting an object (such as a str)
    to a Pyomo solver.

    Parameters
    ----------
    require_available : bool, optional
        True if `available()` method of a standardized solver
        object obtained through `self` must return `True`,
        False otherwise.
    solver_desc : str, optional
        Descriptor for the solver obtained through `self`,
        such as 'local solver'
        or 'global solver'. This argument is used
        for constructing error/exception messages.

    Attributes
    ----------
    require_available
    solver_desc
    """

    def __init__(self, require_available=True, solver_desc='solver'):
        """Initialize self (see class docstring)."""
        self.require_available = require_available
        self.solver_desc = solver_desc

    @staticmethod
    def is_solver_type(obj):
        """
        Return True if object is considered a Pyomo solver,
        False otherwise.

        An object is considered a Pyomo solver provided that
        it has callable attributes named 'solve' and
        'available'.
        """
        return callable(getattr(obj, 'solve', None)) and callable(getattr(obj, 'available', None))

    def __call__(self, obj, require_available=None, solver_desc=None):
        """
        Cast object to a Pyomo solver.

        If `obj` is a string, then ``SolverFactory(obj.lower())``
        is returned. If `obj` is a Pyomo solver type, then
        `obj` is returned.

        Parameters
        ----------
        obj : object
            Object to be cast to Pyomo solver type.
        require_available : bool or None, optional
            True if `available()` method of the resolved solver
            object must return True, False otherwise.
            If `None` is passed, then ``self.require_available``
            is used.
        solver_desc : str or None, optional
            Brief description of the solver, such as 'local solver'
            or 'backup global solver'. This argument is used
            for constructing error/exception messages.
            If `None` is passed, then ``self.solver_desc``
            is used.

        Returns
        -------
        Solver
            Pyomo solver.

        Raises
        ------
        SolverNotResolvable
            If `obj` cannot be cast to a Pyomo solver because
            it is neither a str nor a Pyomo solver type.
        ApplicationError
            In event that solver is not available, the
            method `available(exception_flag=True)` of the
            solver to which `obj` is cast should raise an
            exception of this type. The present method
            will also emit a more detailed error message
            through the default PyROS logger.
        """
        if require_available is None:
            require_available = self.require_available
        if solver_desc is None:
            solver_desc = self.solver_desc
        if isinstance(obj, str):
            solver = SolverFactory(obj.lower())
        elif self.is_solver_type(obj):
            solver = obj
        else:
            raise SolverNotResolvable(f'Cannot cast object `{obj!r}` to a Pyomo optimizer for use as {solver_desc}, as the object is neither a str nor a Pyomo Solver type (got type {type(obj).__name__}).')
        if require_available:
            try:
                solver.available(exception_flag=True)
            except ApplicationError:
                default_pyros_solver_logger.exception(f'Output of `available()` method for {solver_desc} with repr {solver!r} resolved from object {obj} is not `True`. Check solver and any required dependencies have been set up properly.')
                raise
        return solver

    def domain_name(self):
        """Return str briefly describing domain encompassed by self."""
        return 'str or Solver'