from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pickle
from itertools import permutations, product
import logging
from enum import Enum
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp
from pyomo.contrib.doe.scenario import ScenarioGenerator, FiniteDifferenceStep
from pyomo.contrib.doe.result import FisherResults, GridSearchResult
def _direct_kaug(self):
    mod = self.create_model(model_option=ModelOptionLib.parmest)
    if self.discretize_model:
        mod = self.discretize_model(mod, block=False)
    mod.Obj = pyo.Objective(expr=0, sense=pyo.minimize)
    for par in self.param.keys():
        cuid = pyo.ComponentUID(par)
        var = cuid.find_component_on(mod)
        var.setlb(self.param[par])
        var.setub(self.param[par])
    var_name = list(self.param.keys())
    square_result = self._solve_doe(mod, fix=True)
    dsdp_re, col = get_dsdp(mod, list(self.param.keys()), self.param, tee=self.tee_opt)
    dsdp_array = dsdp_re.toarray().T
    self.dsdp = dsdp_array
    self.dsdp = col
    dsdp_extract = []
    measurement_index = []
    for mname in self.measure_name:
        try:
            kaug_no = col.index(mname)
            measurement_index.append(kaug_no)
            dsdp_extract.append(dsdp_array[kaug_no])
        except:
            self.logger.debug('The variable is fixed:  %s', mname)
            zero_sens = np.zeros(len(self.param))
            dsdp_extract.append(zero_sens)
    jac = {}
    for par in self.param.keys():
        jac[par] = []
    for d in range(len(dsdp_extract)):
        for p, par in enumerate(self.param.keys()):
            sensi = dsdp_extract[d][p] * self.scale_constant_value
            if self.scale_nominal_param_value:
                sensi *= self.param[par]
            jac[par].append(sensi)
    if self.specified_prior is None:
        prior_in_use = self.prior_FIM
    else:
        prior_in_use = self.specified_prior
    FIM_analysis = FisherResults(list(self.param.keys()), self.measurement_vars, jacobian_info=None, all_jacobian_info=jac, prior_FIM=prior_in_use, store_FIM=self.FIM_store_name, scale_constant_value=self.scale_constant_value)
    self.jac = jac
    self.mod = mod
    return FIM_analysis