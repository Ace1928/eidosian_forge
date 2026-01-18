import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
def _create_parmest_model(self, data):
    """
        Modify the Pyomo model for parameter estimation
        """
    model = self.model_function(data)
    if len(self.theta_names) == 1 and self.theta_names[0] == 'parmest_dummy_var':
        model.parmest_dummy_var = pyo.Var(initialize=1.0)
    if self.obj_function:
        for obj in model.component_objects(pyo.Objective):
            if obj.name in ['Total_Cost_Objective']:
                raise RuntimeError('Parmest will not override the existing model Objective named ' + obj.name)
            obj.deactivate()
        for expr in model.component_data_objects(pyo.Expression):
            if expr.name in ['FirstStageCost', 'SecondStageCost']:
                raise RuntimeError('Parmest will not override the existing model Expression named ' + expr.name)
        model.FirstStageCost = pyo.Expression(expr=0)
        model.SecondStageCost = pyo.Expression(rule=_SecondStageCostExpr(self.obj_function, data))

        def TotalCost_rule(model):
            return model.FirstStageCost + model.SecondStageCost
        model.Total_Cost_Objective = pyo.Objective(rule=TotalCost_rule, sense=pyo.minimize)
    model = utils.convert_params_to_vars(model, self.theta_names)
    for i, theta in enumerate(self.theta_names):
        var_cuid = ComponentUID(theta)
        var_validate = var_cuid.find_component_on(model)
        if var_validate is None:
            logger.warning('theta_name[%s] (%s) was not found on the model', (i, theta))
        else:
            try:
                var_validate.unfix()
                self.theta_names[i] = repr(var_cuid)
            except:
                logger.warning(theta + ' is not a variable')
    self.parmest_model = model
    return model