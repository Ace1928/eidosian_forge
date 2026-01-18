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
def _add_sensitivity_data(self, param_list):
    block = self.block
    sens_data_list = block._sens_data_list
    for i, comp in enumerate(param_list):
        if comp.ctype is Param:
            parent = comp.parent_component()
            if not parent.mutable:
                raise ValueError('Parameters within paramList must be mutable. Got %s, which is not mutable.' % comp.name)
            if comp.is_indexed():
                d = {k: value(comp[k]) for k in comp.index_set()}
                var = Var(comp.index_set(), initialize=d)
            else:
                d = value(comp)
                var = Var(initialize=d)
            name = self.get_default_var_name(parent.local_name)
            name = unique_component_name(block, name)
            block.add_component(name, var)
            if comp.is_indexed():
                sens_data_list.extend(((var[idx], param, i, idx) for idx, param in _generate_component_items(comp)))
            else:
                sens_data_list.append((var, comp, i, _NotAnIndex))
        elif comp.ctype is Var:
            parent = comp.parent_component()
            for _, data in _generate_component_items(comp):
                if not data.fixed:
                    raise ValueError('Specified "parameter" variables must be fixed. Got %s, which is not fixed.' % comp.name)
            if comp.is_indexed():
                d = {k: value(comp[k]) for k in comp.index_set()}
                param = Param(comp.index_set(), mutable=True, initialize=d)
            else:
                d = value(comp)
                param = Param(mutable=True, initialize=d)
            name = self.get_default_param_name(parent.local_name)
            name = unique_component_name(block, name)
            block.add_component(name, param)
            if comp.is_indexed():
                sens_data_list.extend(((var, param[idx], i, idx) for idx, var in _generate_component_items(comp)))
            else:
                sens_data_list.append((comp, param, i, _NotAnIndex))