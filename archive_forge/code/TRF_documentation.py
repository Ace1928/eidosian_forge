import logging
from pyomo.core.base.range import NumericRange
from pyomo.common.config import (
from pyomo.contrib.trustregion.filter import Filter, FilterElement
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.util import IterationLogger
from pyomo.opt import SolverFactory

        This method calls the TRF algorithm.

        Parameters
        ----------
        model : ConcreteModel
            The model to be solved using the Trust Region Framework.
        degrees_of_freedom_variables : List[Var]
            User-supplied input. The user must provide a list of vars which
            are the degrees of freedom or decision variables within
            the model.
        ext_fcn_surrogate_map_rule : Function, optional
            In the 2020 Yoshio/Biegler paper, this is referred to as
            the basis function `b(w)`.
            This is the low-fidelity model with which to solve the original
            process model problem and which is integrated into the
            surrogate model.
            The default is 0 (i.e., no basis function rule.)

        