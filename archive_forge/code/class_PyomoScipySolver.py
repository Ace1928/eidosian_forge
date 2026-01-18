from collections import namedtuple
from pyomo.core.base.objective import Objective
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.opt import SolverResults, TerminationCondition
from pyomo.common.dependencies import (
class PyomoScipySolver(object):

    def __init__(self, options=None):
        if options is None:
            options = {}
        self._nlp = None
        self._nlp_solver = None
        self._full_output = None
        self.options = options

    def available(self, exception_flag=False):
        return bool(numpy_available and scipy_available)

    def license_is_valid(self):
        return True

    def version(self):
        return tuple((int(_) for _ in sp.__version__.split('.')))

    def set_options(self, options):
        self.options = options

    def solve(self, model, timer=None, tee=False):
        """
        Parameters
        ----------
        model: BlockData
            The model that will be solved
        timer: HierarchicalTimer
            A HierarchicalTimer that "sub-timers" created by this object
            will be attached to. If not provided, a new timer is created.
        tee: Bool
            A dummy flag indicating whether solver output should be displayed.
            The current SciPy solvers supported have no output, so setting this
            flag does not do anything.

        Returns
        -------
        SolverResults
            Contains the results of the solve

        """
        if timer is None:
            timer = HierarchicalTimer()
        self._timer = timer
        self._timer.start('solve')
        active_objs = list(model.component_data_objects(Objective, active=True))
        if len(active_objs) == 0:
            obj_name = unique_component_name(model, '_obj')
            obj = Objective(expr=0.0)
            model.add_component(obj_name, obj)
        nlp = pyomo_nlp.PyomoNLP(model)
        self._nlp = nlp
        if len(active_objs) == 0:
            model.del_component(obj_name)
        self._nlp_solver = self.create_nlp_solver(options=self.options)
        x0 = nlp.get_primals()
        results = self._nlp_solver.solve(x0=x0)
        for var, val in zip(nlp.get_pyomo_variables(), nlp.get_primals()):
            var.set_value(val)
        self._timer.stop('solve')
        pyomo_results = self.get_pyomo_results(model, results)
        return pyomo_results

    def get_nlp(self):
        return self._nlp

    def create_nlp_solver(self, **kwds):
        raise NotImplementedError('%s has not implemented the create_nlp_solver method' % self.__class__)

    def get_pyomo_results(self, model, scipy_results):
        raise NotImplementedError('%s has not implemented the get_results method' % self.__class__)

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass