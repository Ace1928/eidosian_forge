from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base.external import ExternalFunction
from pyomo.network import Port
from pyomo.common.collections import ComponentSet
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
from pyomo.opt import SolverFactory
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.core.util import target_list
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from weakref import ref as weakref_ref
from math import floor
import logging
def _transform_disjunctionData(self, obj, idx, transBlock=None, transformed_parent_disjunct=None):
    if not obj.active:
        return
    if len(obj.disjuncts) == 0:
        raise GDP_Error("Disjunction '%s' is empty. This is likely indicative of a modeling error." % obj.getname(fully_qualified=True))
    if transBlock is None and transformed_parent_disjunct is not None:
        transBlock = self._get_transformation_block(transformed_parent_disjunct)
    if transBlock is None:
        transBlock = self._get_transformation_block(obj.parent_block())
    variable_partitions = self.variable_partitions
    partition_method = self.partitioning_method
    partition = variable_partitions.get(obj)
    if partition is None:
        partition = variable_partitions.get(None)
        if partition is None:
            method = partition_method.get(obj)
            if method is None:
                method = partition_method.get(None)
            method = method if method is not None else arbitrary_partition
            if self._config.num_partitions is None:
                P = None
            else:
                P = self._config.num_partitions.get(obj)
                if P is None:
                    P = self._config.num_partitions.get(None)
            if P is None:
                raise GDP_Error('No value for P was given for disjunction %s! Please specify a value of P (number of partitions), if you do not specify the partitions directly.' % obj.name)
            partition = method(obj, P)
    partition = [ComponentSet(var_list) for var_list in partition]
    transformed_disjuncts = []
    for disjunct in obj.disjuncts:
        transformed_disjunct = self._transform_disjunct(disjunct, partition, transBlock)
        if transformed_disjunct is not None:
            transformed_disjuncts.append(transformed_disjunct)
            transBlock.indicator_var_equalities[len(transBlock.indicator_var_equalities)] = disjunct.indicator_var.equivalent_to(transformed_disjunct.indicator_var)
    transformed_disjunction = Disjunction(expr=[disj for disj in transformed_disjuncts])
    transBlock.add_component(unique_component_name(transBlock, obj.getname(fully_qualified=True)), transformed_disjunction)
    obj._algebraic_constraint = weakref_ref(transformed_disjunction)
    obj.deactivate()