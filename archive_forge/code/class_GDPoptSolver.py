from pyomo.common.config import document_kwargs_from_configdict, ConfigDict
from pyomo.contrib.gdpopt import __version__
from pyomo.contrib.gdpopt.config_options import (
from pyomo.opt.base import SolverFactory
@SolverFactory.register('gdpopt', doc='The GDPopt decomposition-based Generalized Disjunctive Programming (GDP) solver')
class GDPoptSolver(object):
    """Decomposition solver for Generalized Disjunctive Programming (GDP)
    problems.

    The GDPopt (Generalized Disjunctive Programming optimizer) solver applies a
    variety of decomposition-based approaches to solve Generalized Disjunctive
    Programming (GDP) problems. GDP models can include nonlinear, continuous
    variables and constraints, as well as logical conditions.

    These approaches include:

    - Logic-based outer approximation (LOA)
    - Logic-based branch-and-bound (LBB)
    - Partial surrogate cuts [pending]
    - Generalized Bender decomposition [pending]

    This solver implementation was developed by Carnegie Mellon University in
    the research group of Ignacio Grossmann.

    For nonconvex problems, LOA may not report rigorous lower/upper bounds.

    Questions: Please make a post at StackOverflow and/or contact Qi Chen
    <https://github.com/qtothec> or David Bernal <https://github.com/bernalde>.

    Several key GDPopt components were prototyped by BS and MS students:

    - Logic-based branch and bound: Sunjeev Kale
    - MC++ interface: Johnny Bates
    - LOA set-covering initialization: Eloy Fernandez
    - Logic-to-linear transformation: Romeo Valentin

    """
    CONFIG = ConfigDict('GDPopt')
    _add_common_configs(CONFIG)

    @document_kwargs_from_configdict(CONFIG)
    def solve(self, model, **kwds):
        """Solve the model.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        options = kwds.pop('options', {})
        config = self.CONFIG(options, preserve_implicit=True)
        config.set_value(kwds, skip_implicit=True)
        alg_config = _get_algorithm_config()(options, preserve_implicit=True)
        alg_config.set_value(kwds, skip_implicit=True)
        _handle_strategy_deprecation(alg_config)
        algorithm = alg_config.algorithm
        if algorithm is None:
            raise ValueError('No algorithm was specified to the solve method. Please specify an algorithm or use an algorithm-specific solver.')
        kwds.pop('algorithm', None)
        kwds.pop('strategy', None)
        return SolverFactory(_supported_algorithms[algorithm][0]).solve(model, **kwds)

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        """Solver is always available. Though subsolvers may not be, they will
        raise an error when the time comes.
        """
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__
    _metasolver = False