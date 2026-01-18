import logging
from pyomo.core.base.range import NumericRange
from pyomo.common.config import (
from pyomo.contrib.trustregion.filter import Filter, FilterElement
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.util import IterationLogger
from pyomo.opt import SolverFactory
@SolverFactory.register('trustregion', doc='Trust region algorithm "solver" for black box/glass box optimization')
class TrustRegionSolver(object):
    """
    The Trust Region Solver is a 'solver' based on the 2016/2018/2020 AiChE
    papers by Eason (2016/2018), Yoshio (2020), and Biegler.

    """
    CONFIG = _trf_config()

    def __init__(self, **kwds):
        self.config = self.CONFIG(kwds)

    def available(self, exception_flag=True):
        """
        Check if solver is available.
        """
        return True

    def version(self):
        """
        Return a 3-tuple describing the solver version.
        """
        return __version__

    def license_is_valid(self):
        """
        License for using Trust Region solver.
        """
        return True

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        pass

    @document_kwargs_from_configdict(CONFIG)
    def solve(self, model, degrees_of_freedom_variables, ext_fcn_surrogate_map_rule=None, **kwds):
        """
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

        """
        config = self.config(kwds.pop('options', {}))
        config.set_value(kwds)
        if ext_fcn_surrogate_map_rule is None:
            ext_fcn_surrogate_map_rule = lambda comp, ef: 0
        result = trust_region_method(model, degrees_of_freedom_variables, ext_fcn_surrogate_map_rule, config)
        return result