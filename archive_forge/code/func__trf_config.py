import logging
from pyomo.core.base.range import NumericRange
from pyomo.common.config import (
from pyomo.contrib.trustregion.filter import Filter, FilterElement
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.util import IterationLogger
from pyomo.opt import SolverFactory
def _trf_config():
    """
    Generate the configuration dictionary.
    The user may change the configuration options during the instantiation
    of the trustregion solver:
        >>> optTRF = SolverFactory('trustregion',
        ...                        solver='ipopt',
        ...                        maximum_iterations=50,
        ...                        minimum_radius=1e-5,
        ...                        verbose=True)

    The user may also update the configuration after instantiation:
        >>> optTRF = SolverFactory('trustregion')
        >>> optTRF._CONFIG.trust_radius = 0.5

    The user may also update the configuration as part of the solve call:
        >>> optTRF = SolverFactory('trustregion')
        >>> optTRF.solve(model, decision_variables, trust_radius=0.5)
    Returns
    -------
    CONFIG : ConfigDict
        This holds all configuration options to be passed to the TRF solver.

    """
    CONFIG = ConfigDict('TrustRegion')
    CONFIG.declare('solver', ConfigValue(default='ipopt', description='Solver to use. Default = ``ipopt``.'))
    CONFIG.declare('keepfiles', ConfigValue(default=False, domain=Bool, description='Optional. Whether or not to write files of sub-problems for use in debugging. Default = False.'))
    CONFIG.declare('tee', ConfigValue(default=False, domain=Bool, description='Optional. Sets the ``tee`` for sub-solver(s) utilized. Default = False.'))
    CONFIG.declare('verbose', ConfigValue(default=False, domain=Bool, description="Optional. When True, print each iteration's relevant information to the console as well as to the log. Default = False."))
    CONFIG.declare('trust_radius', ConfigValue(default=1.0, domain=PositiveFloat, description='Initial trust region radius ``delta_0``. Default = 1.0.'))
    CONFIG.declare('minimum_radius', ConfigValue(default=1e-06, domain=PositiveFloat, description='Minimum allowed trust region radius ``delta_min``. Default = 1e-6.'))
    CONFIG.declare('maximum_radius', ConfigValue(default=CONFIG.trust_radius * 100, domain=PositiveFloat, description='Maximum allowed trust region radius. If trust region radius reaches maximum allowed, solver will exit. Default = 100 * trust_radius.'))
    CONFIG.declare('maximum_iterations', ConfigValue(default=50, domain=PositiveInt, description='Maximum allowed number of iterations. Default = 50.'))
    CONFIG.declare('feasibility_termination', ConfigValue(default=1e-05, domain=PositiveFloat, description='Feasibility measure termination tolerance ``epsilon_theta``. Default = 1e-5.'))
    CONFIG.declare('step_size_termination', ConfigValue(default=CONFIG.feasibility_termination, domain=PositiveFloat, description='Step size termination tolerance ``epsilon_s``. Matches the feasibility termination tolerance by default.'))
    CONFIG.declare('minimum_feasibility', ConfigValue(default=0.0001, domain=PositiveFloat, description='Minimum feasibility measure ``theta_min``. Default = 1e-4.'))
    CONFIG.declare('switch_condition_kappa_theta', ConfigValue(default=0.1, domain=In(NumericRange(0, 1, 0, (False, False))), description='Switching condition parameter ``kappa_theta``. Contained in open set (0, 1). Default = 0.1.'))
    CONFIG.declare('switch_condition_gamma_s', ConfigValue(default=2.0, domain=PositiveFloat, description='Switching condition parameter ``gamma_s``. Must satisfy: ``gamma_s > 1/(1+mu)`` where ``mu`` is contained in set (0, 1]. Default = 2.0.'))
    CONFIG.declare('radius_update_param_gamma_c', ConfigValue(default=0.5, domain=In(NumericRange(0, 1, 0, (False, False))), description='Lower trust region update parameter ``gamma_c``. Default = 0.5.'))
    CONFIG.declare('radius_update_param_gamma_e', ConfigValue(default=2.5, domain=In(NumericRange(1, None, 0)), description='Upper trust region update parameter ``gamma_e``. Default = 2.5.'))
    CONFIG.declare('ratio_test_param_eta_1', ConfigValue(default=0.05, domain=In(NumericRange(0, 1, 0, (False, False))), description='Lower ratio test parameter ``eta_1``. Must satisfy: ``0 < eta_1 <= eta_2 < 1``. Default = 0.05.'))
    CONFIG.declare('ratio_test_param_eta_2', ConfigValue(default=0.2, domain=In(NumericRange(0, 1, 0, (False, False))), description='Lower ratio test parameter ``eta_2``. Must satisfy: ``0 < eta_1 <= eta_2 < 1``. Default = 0.2.'))
    CONFIG.declare('maximum_feasibility', ConfigValue(default=50.0, domain=PositiveFloat, description='Maximum allowable feasibility measure ``theta_max``. Parameter for use in filter method.Default = 50.0.'))
    CONFIG.declare('param_filter_gamma_theta', ConfigValue(default=0.01, domain=In(NumericRange(0, 1, 0, (False, False))), description='Fixed filter parameter ``gamma_theta`` within (0, 1). Default = 0.01'))
    CONFIG.declare('param_filter_gamma_f', ConfigValue(default=0.01, domain=In(NumericRange(0, 1, 0, (False, False))), description='Fixed filter parameter ``gamma_f`` within (0, 1). Default = 0.01'))
    return CONFIG