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
class PositiveIntOrMinusOne:
    """
    Domain validator for objects castable to a
    strictly positive int or -1.
    """

    def __call__(self, obj):
        """
        Cast object to positive int or -1.

        Parameters
        ----------
        obj : object
            Object of interest.

        Returns
        -------
        int
            Positive int, or -1.

        Raises
        ------
        ValueError
            If object not castable to positive int, or -1.
        """
        ans = int(obj)
        if ans != float(obj) or (ans <= 0 and ans != -1):
            raise ValueError(f'Expected positive int or -1, but received value {obj!r}')
        return ans

    def domain_name(self):
        """Return str briefly describing domain encompassed by self."""
        return 'positive int or -1'