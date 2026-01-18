import pyomo.core.expr as EXPR
from pyomo.core import (
from pyomo.core.base.misc import create_name
from pyomo.core.plugins.transform.util import partial
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.util import collectAbstractComponents
import logging
def _walk_expr(expr, varMap):
    """
    Walks an expression tree, making the replacements defined in varMap
    """
    visitor = VarmapVisitor(varMap)
    return visitor.dfs_postorder_stack(expr)