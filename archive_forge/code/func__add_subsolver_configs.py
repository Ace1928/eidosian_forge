import logging
from pyomo.common.config import (
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger
from pyomo.common.deprecation import deprecation_warning
def _add_subsolver_configs(CONFIG):
    """Adds the subsolver-related configurations.

    Parameters
    ----------
    CONFIG : ConfigBlock
        The specific configurations for MindtPy.
    """
    CONFIG.declare('nlp_solver', ConfigValue(default='ipopt', domain=In(['ipopt', 'appsi_ipopt', 'gams', 'baron', 'cyipopt']), description='NLP subsolver name', doc='Which NLP subsolver is going to be used for solving the nonlinearsubproblems.'))
    CONFIG.declare('nlp_solver_args', ConfigBlock(implicit=True, description='NLP subsolver options', doc='Which NLP subsolver options to be passed to the solver while solving the nonlinear subproblems.'))
    CONFIG.declare('mip_solver', ConfigValue(default='glpk', domain=In(['gurobi', 'cplex', 'cbc', 'glpk', 'gams', 'gurobi_persistent', 'cplex_persistent', 'appsi_cplex', 'appsi_gurobi']), description='MIP subsolver name', doc='Which MIP subsolver is going to be used for solving the mixed-integer main problems.'))
    CONFIG.declare('mip_solver_args', ConfigBlock(implicit=True, description='MIP subsolver options', doc='Which MIP subsolver options to be passed to the solver while solving the mixed-integer main problems.'))
    CONFIG.declare('mip_solver_mipgap', ConfigValue(default=0.0001, domain=PositiveFloat, description='Mipgap passed to MIP solver.'))
    CONFIG.declare('threads', ConfigValue(default=0, domain=NonNegativeInt, description='Threads', doc='Threads used by MIP solver and NLP solver.'))
    CONFIG.declare('regularization_mip_threads', ConfigValue(default=0, domain=NonNegativeInt, description='regularization MIP threads', doc='Threads used by MIP solver to solve regularization main problem.'))
    CONFIG.declare('solver_tee', ConfigValue(default=False, description='Stream the output of MIP solver and NLP solver to terminal.', domain=bool))
    CONFIG.declare('mip_solver_tee', ConfigValue(default=False, description='Stream the output of MIP solver to terminal.', domain=bool))
    CONFIG.declare('nlp_solver_tee', ConfigValue(default=False, description='Stream the output of nlp solver to terminal.', domain=bool))
    CONFIG.declare('mip_regularization_solver', ConfigValue(default=None, domain=In(['gurobi', 'cplex', 'cbc', 'glpk', 'gams', 'gurobi_persistent', 'cplex_persistent', 'appsi_cplex', 'appsi_gurobi']), description='MIP subsolver for regularization problem', doc='Which MIP subsolver is going to be used for solving the regularization problem.'))