import time
import pyomo.opt
from pyomo.opt import SolverFactory
from pyomo.core import TransformationFactory
from pyomo.common.collections import Bunch
@SolverFactory.register('mpec_minlp', doc='MPEC solver transforms to a MINLP')
class MPEC_Solver2(pyomo.opt.OptSolver):

    def __init__(self, **kwds):
        kwds['type'] = 'mpec_minlp'
        pyomo.opt.OptSolver.__init__(self, **kwds)
        self._metasolver = True

    def _presolve(self, *args, **kwds):
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _apply_solver(self):
        start_time = time.time()
        xfrm = TransformationFactory('mpec.simple_disjunction')
        xfrm.apply_to(self._instance)
        xfrm = TransformationFactory('gdp.bigm')
        xfrm.apply_to(self._instance, bigM=self.options.get('bigM', 10 ** 6))
        solver = self.options.solver
        if not self.options.solver:
            self.options.solver = solver = 'glpk'
        with pyomo.opt.SolverFactory(solver) as opt:
            self.results = opt.solve(self._instance, tee=self._tee, timelimit=self._timelimit)
            from pyomo.mpec import Complementarity
            for cuid in self._instance._transformation_data['mpec.simple_disjunction'].compl_cuids:
                cobj = cuid.find_component_on(self._instance)
                cobj.parent_block().reclassify_component_type(cobj, Complementarity)
            stop_time = time.time()
            self.wall_time = stop_time - start_time
            return Bunch(rc=getattr(opt, '_rc', None), log=getattr(opt, '_log', None))

    def _postsolve(self):
        solv = self.results.solver
        solv.name = self.options.subsolver
        solv.wallclock_time = self.wall_time
        self._instance.solutions.store_to(self.results)
        self._instance = None
        return self.results