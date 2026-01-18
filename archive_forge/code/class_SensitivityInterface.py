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
class SensitivityInterface(object):

    def __init__(self, instance, clone_model=True):
        """Constructor clones model if necessary and attaches
        to this object.
        """
        self._original_model = instance
        if clone_model:
            self.model_instance = instance.clone()
        else:
            self.model_instance = instance

    @classmethod
    def get_default_block_name(self):
        return '_SENSITIVITY_TOOLBOX_DATA'

    @staticmethod
    def get_default_var_name(name):
        return name

    @staticmethod
    def get_default_param_name(name):
        return name

    def _process_param_list(self, paramList):
        orig = self._original_model
        instance = self.model_instance
        if orig is not instance:
            paramList = list((ComponentUID(param, context=orig).find_component_on(instance) for param in paramList))
        return paramList

    def _add_data_block(self, existing_block=None):
        if existing_block is not None:
            if hasattr(existing_block, '_has_replaced_expressions') and (not existing_block._has_replaced_expressions):
                for var, _, _, _ in existing_block._sens_data_list:
                    var.fix()
                self.model_instance.del_component(existing_block)
            else:
                msg = 'Re-using sensitivity interface is not supported when calculating sensitivity for mutable parameters. Used fixed vars instead if you want to do this.'
                raise RuntimeError(msg)
        block = Block()
        self.model_instance.add_component(self.get_default_block_name(), block)
        self.block = block
        block._has_replaced_expressions = False
        block._sens_data_list = []
        block._paramList = None
        block.constList = ConstraintList()
        return block

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

    def setup_sensitivity(self, paramList):
        instance = self.model_instance
        paramList = self._process_param_list(paramList)
        existing_block = instance.component(self.get_default_block_name())
        block = self._add_data_block(existing_block=existing_block)
        block._sens_data_list = []
        block._paramList = paramList
        self._add_sensitivity_data(paramList)
        sens_data_list = block._sens_data_list
        for var, _, _, _ in sens_data_list:
            var.unfix()
        variableSubMap = dict(((id(param), var) for var, param, list_idx, _ in sens_data_list if paramList[list_idx].ctype is Param))
        if variableSubMap:
            block._replaced_map = self._replace_parameters_in_constraints(variableSubMap)
            block._has_replaced_expressions = True
        block.paramConst = ConstraintList()
        for var, param, _, _ in sens_data_list:
            block.paramConst.add(var - param == 0)
        _add_sensitivity_suffixes(instance)
        for i, (var, _, _, _) in enumerate(sens_data_list):
            idx = i + 1
            con = block.paramConst[idx]
            instance.sens_state_0[var] = idx
            instance.sens_state_1[var] = idx
            instance.sens_init_constr[con] = idx
            instance.dcdp[con] = idx

    def perturb_parameters(self, perturbList):
        instance = self.model_instance
        sens_data_list = self.block._sens_data_list
        paramConst = self.block.paramConst
        if len(self.block._paramList) != len(perturbList):
            raise ValueError('Length of paramList argument does not equal length of perturbList')
        for i, (var, param, list_idx, comp_idx) in enumerate(sens_data_list):
            con = paramConst[i + 1]
            if comp_idx is _NotAnIndex:
                ptb = value(perturbList[list_idx])
            else:
                try:
                    ptb = value(perturbList[list_idx][comp_idx])
                except TypeError:
                    ptb = value(perturbList[list_idx])
            instance.sens_state_value_1[var] = ptb
            instance.DeltaP[con] = value(var - ptb)