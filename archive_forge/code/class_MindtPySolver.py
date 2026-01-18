from pyomo.contrib.mindtpy import __version__
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_config
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.contrib.mindtpy.config_options import _supported_algorithms
@SolverFactory.register('mindtpy', doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo')
class MindtPySolver(object):
    """
    Decomposition solver for Mixed-Integer Nonlinear Programming (MINLP) problems.

    The MindtPy (Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo) solver
    applies a variety of decomposition-based approaches to solve Mixed-Integer
    Nonlinear Programming (MINLP) problems.
    These approaches include:

    - Outer approximation (OA)
    - Global outer approximation (GOA)
    - Regularized outer approximation (ROA)
    - LP/NLP based branch-and-bound (LP/NLP)
    - Global LP/NLP based branch-and-bound (GLP/NLP)
    - Regularized LP/NLP based branch-and-bound (RLP/NLP)
    - Feasibility pump (FP)
    """
    CONFIG = _get_MindtPy_config()

    def available(self, exception_flag=True):
        """Check if solver is available."""
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    @document_kwargs_from_configdict(CONFIG)
    def solve(self, model, **kwds):
        """Solve the model.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        options = kwds.pop('options', {})
        config = self.CONFIG(options, preserve_implicit=True)
        config.set_value(kwds, skip_implicit=True)
        return SolverFactory(_supported_algorithms[config.strategy][0]).solve(model, **kwds)

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass