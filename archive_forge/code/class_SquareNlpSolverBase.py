from collections import namedtuple
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigBlock
from pyomo.util.subsystems import create_subsystem_block
class SquareNlpSolverBase(object):
    """A base class for NLP solvers that act on a square system
    of equality constraints.

    """
    OPTIONS = ConfigBlock()

    def __init__(self, nlp, timer=None, options=None):
        """
        Arguments
        ---------
        nlp: ExtendedNLP
            An instance of ExtendedNLP that will be solved.
            ExtendedNLP is required to ensure that the NLP has equal
            numbers of primal variables and equality constraints.

        """
        if timer is None:
            timer = HierarchicalTimer()
        if options is None:
            options = {}
        self.options = self.OPTIONS(options)
        self._timer = timer
        self._nlp = nlp
        self._function_values = None
        self._jacobian = None
        if self._nlp.n_eq_constraints() != self._nlp.n_primals():
            raise RuntimeError('Cannot construct a square solver for an NLP that does not have the same numbers of variables as equality constraints. Got %s variables and %s equalities.' % (self._nlp.n_primals(), self._nlp.n_eq_constraints()))

    def solve(self, x0=None):
        raise NotImplementedError('%s has not implemented the solve method' % self.__class__)

    def evaluate_function(self, x0):
        self._nlp.set_primals(x0)
        values = self._nlp.evaluate_eq_constraints()
        return values

    def evaluate_jacobian(self, x0):
        self._nlp.set_primals(x0)
        self._jacobian = self._nlp.evaluate_jacobian_eq(out=self._jacobian)
        return self._jacobian