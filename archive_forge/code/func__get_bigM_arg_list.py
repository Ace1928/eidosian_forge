from pyomo.gdp import GDP_Error
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.contrib.fbbt.interval as interval
from pyomo.core import Suffix
def _get_bigM_arg_list(self, bigm_args, block):
    arg_list = []
    if bigm_args is None:
        return arg_list
    while block is not None:
        if block in bigm_args:
            arg_list.append({block: bigm_args[block]})
        block = block.parent_block()
    return arg_list