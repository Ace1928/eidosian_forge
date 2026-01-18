from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.expr import identify_variables
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import is_child_of, get_gdp_tree
from pyomo.repn.standard_repn import generate_standard_repn
import logging
def _create_transformation_constraints(self, root_disjunction, bound_dict, gdp_forest, transformation_blocks):
    to_transform = ComponentSet([root_disjunction])
    while to_transform:
        disjunction = to_transform.pop()
        trans_block = self._add_transformation_block(disjunction, transformation_blocks)
        if self.transformation_name not in disjunction._transformation_map:
            disjunction._transformation_map[self.transformation_name] = ComponentMap()
        trans_map = disjunction._transformation_map[self.transformation_name]
        for disj in disjunction.disjuncts:
            to_transform.update(gdp_forest.children(disj))
        for v, v_bounds in bound_dict.items():
            unique_id = len(trans_block.transformed_bound_constraints)
            if not any((disj in v_bounds for disj in disjunction.disjuncts)):
                continue
            all_lbs = True
            all_ubs = True
            lb_expr = 0
            ub_expr = 0
            deactivate_lower = ComponentSet()
            deactivate_upper = ComponentSet()
            for disj in disjunction.disjuncts:
                lb, ub = self._get_tightest_ancestral_bounds(v_bounds, disj, gdp_forest)
                if lb is None:
                    all_lbs = False
                    if not all_ubs:
                        break
                if ub is None:
                    all_ubs = False
                    if not all_lbs:
                        break
                if all_lbs:
                    lb_expr += lb * disj.binary_indicator_var
                    if disj in v_bounds['to_deactivate']:
                        deactivate_lower.update(v_bounds['to_deactivate'][disj])
                if all_ubs:
                    ub_expr += ub * disj.binary_indicator_var
                    if disj in v_bounds['to_deactivate']:
                        deactivate_upper.update(v_bounds['to_deactivate'][disj])
            if all_lbs:
                idx = (v.local_name + '_lb', unique_id)
                trans_block.transformed_bound_constraints[idx] = lb_expr <= v
                trans_map[v] = [trans_block.transformed_bound_constraints[idx]]
                for c in deactivate_lower:
                    if c.lower is None:
                        continue
                    if c.upper is None or (all_ubs and c in deactivate_upper):
                        c.deactivate()
                    else:
                        c.deactivate()
                        c.parent_block().add_component(unique_component_name(c.parent_block(), c.local_name + '_ub'), Constraint(expr=v <= c.upper))
            if all_ubs:
                idx = (v.local_name + '_ub', unique_id + 1)
                trans_block.transformed_bound_constraints[idx] = ub_expr >= v
                if v in trans_map:
                    trans_map[v].append(trans_block.transformed_bound_constraints[idx])
                else:
                    trans_map[v] = [trans_block.transformed_bound_constraints[idx]]
                for c in deactivate_upper:
                    if c.upper is None:
                        continue
                    if c.lower is None or (all_lbs and c in deactivate_lower):
                        c.deactivate()
                    else:
                        c.deactivate()
                        c.parent_block().add_component(unique_component_name(c.parent_block(), c.local_name + '_lb'), Constraint(expr=v >= c.lower))