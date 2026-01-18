from collections import namedtuple
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigBlock
from pyomo.util.subsystems import create_subsystem_block
class ScalarDenseSquareNlpSolver(DenseSquareNlpSolver):

    def __init__(self, nlp, timer=None, options=None):
        super().__init__(nlp, timer=timer, options=options)
        if nlp.n_primals() != 1:
            raise RuntimeError('Cannot use the scipy.optimize.newton solver on an NLP with more than one variable and equality constraint. Got %s primals. Please use RootNlpSolver or FsolveNlpSolver instead.')