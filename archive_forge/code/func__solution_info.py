from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
def _solution_info(self, m, dv_set):
    """
        Solution information. Only for optimization problem

        Parameters
        ----------
        m: model
        dv_set: design variable dictionary

        Returns
        -------
        model_info: model solutions dictionary containing the following key:value pairs
            -['obj']: a scalar number of objective function value
            -['det']: a scalar number of determinant calculated by the model (different from FIM_info['det'] which
            is calculated by numpy)
            -['trace']: a scalar number of trace calculated by the model
            -[design variable name]: a list of design variable solution
        """
    self.obj_value = value(m.obj)
    if self.obj == 'det':
        self.obj_det = np.exp(value(m.obj)) / self.fim_scale_constant_value ** len(self.parameter_names)
    elif self.obj == 'trace':
        self.obj_trace = np.exp(value(m.obj)) / self.fim_scale_constant_value
    design_variable_names = list(dv_set.keys())
    dv_times = list(dv_set.values())
    solution = {}
    for d, dname in enumerate(design_variable_names):
        sol = []
        if dv_times[d] is not None:
            for t, time in enumerate(dv_times[d]):
                newvar = getattr(m, dname)[time]
                sol.append(value(newvar))
        else:
            newvar = getattr(m, dname)
            sol.append(value(newvar))
        solution[dname] = sol
    self.solution = solution