import inspect
import logging
import sys
from copy import deepcopy
from collections import deque
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError, TemplateExpressionError
from pyomo.common.numeric_types import (
import pyomo.core.expr.expr_common as common
from pyomo.core.expr.symbol_map import SymbolMap
def _process_node_bx(self, node, recursion_limit):
    """Recursive routine for processing nodes with only 'bx' callbacks

        This is a special-case implementation of the "general"
        StreamBasedExpressionVisitor node processor for the case that
        only beforeChild and exitNode are defined (see also the
        definition of the client_methods dict).

        """
    if not recursion_limit:
        recursion_limit = self._compute_actual_recursion_limit()
    else:
        recursion_limit -= 1
    if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
        args = ()
    else:
        args = node.args
    data = []
    try:
        child_idx = -1
        arg_iter = iter(args)
        for child in arg_iter:
            child_idx += 1
            tmp = self.beforeChild(node, child, child_idx)
            if tmp is None:
                descend = True
            else:
                descend, child_result = tmp
            if descend:
                data.append(self._process_node(child, recursion_limit))
            else:
                data.append(child_result)
    except RevertToNonrecursive:
        self._recursive_frame_to_nonrecursive_stack(locals())
        raise
    finally:
        pass
    return self.exitNode(node, data)