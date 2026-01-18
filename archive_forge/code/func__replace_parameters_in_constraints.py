from pyomo.environ import (
from pyomo.common.sorting import sorted_robust
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface, InTempDir
import logging
import os
import io
import shutil
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy, scipy_available
def _replace_parameters_in_constraints(self, variableSubMap):
    instance = self.model_instance
    block = self.block
    param_replacer = ExpressionReplacementVisitor(substitute=variableSubMap, remove_named_expressions=True)
    new_old_comp_map = ComponentMap()
    for obj in list(instance.component_data_objects(Objective, active=True, descend_into=True)):
        tempName = unique_component_name(block, obj.local_name)
        new_expr = param_replacer.walk_expression(obj.expr)
        block.add_component(tempName, Objective(expr=new_expr))
        new_old_comp_map[block.component(tempName)] = obj
        obj.deactivate()
    old_con_list = list(instance.component_data_objects(Constraint, active=True, descend_into=True))
    last_idx = 0
    for con in old_con_list:
        if con.equality or con.lower is None or con.upper is None:
            new_expr = param_replacer.walk_expression(con.expr)
            block.constList.add(expr=new_expr)
            last_idx += 1
            new_old_comp_map[block.constList[last_idx]] = con
        else:
            new_body = param_replacer.walk_expression(con.body)
            new_lower = param_replacer.walk_expression(con.lower)
            new_upper = param_replacer.walk_expression(con.upper)
            block.constList.add(expr=new_lower <= new_body)
            last_idx += 1
            new_old_comp_map[block.constList[last_idx]] = con
            block.constList.add(expr=new_body <= new_upper)
            last_idx += 1
            new_old_comp_map[block.constList[last_idx]] = con
        con.deactivate()
    return new_old_comp_map