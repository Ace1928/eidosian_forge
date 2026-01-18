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
def _nonrecursive_walker_loop(self, ptr):
    _, node, args, _, data, child_idx = ptr
    try:
        while 1:
            if child_idx < ptr[3]:
                child_idx += 1
                child = ptr[2][child_idx]
                if self.beforeChild is not None:
                    tmp = self.beforeChild(node, child, child_idx)
                    if tmp is None:
                        descend = True
                        child_result = None
                    else:
                        descend, child_result = tmp
                    if not descend:
                        if self.acceptChildResult is not None:
                            data = self.acceptChildResult(node, data, child_result, child_idx)
                        elif data is not None:
                            data.append(child_result)
                        if self.afterChild is not None:
                            self.afterChild(node, child, child_idx)
                        continue
                ptr = ptr[:4] + (data, child_idx)
                if self.enterNode is not None:
                    tmp = self.enterNode(child)
                    if tmp is None:
                        args = data = None
                    else:
                        args, data = tmp
                else:
                    args = None
                    data = []
                if args is None:
                    if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                        args = ()
                    else:
                        args = child.args
                if hasattr(args, '__enter__'):
                    args.__enter__()
                node = child
                child_idx = -1
                ptr = (ptr, node, args, len(args) - 1, data, child_idx)
            else:
                if hasattr(ptr[2], '__exit__'):
                    ptr[2].__exit__(None, None, None)
                if self.exitNode is not None:
                    node_result = self.exitNode(node, data)
                else:
                    node_result = data
                ptr = ptr[0]
                if ptr is None:
                    if self.finalizeResult is not None:
                        return self.finalizeResult(node_result)
                    else:
                        return node_result
                node, child = (ptr[1], node)
                data = ptr[4]
                child_idx = ptr[5]
                if self.acceptChildResult is not None:
                    data = self.acceptChildResult(node, data, node_result, child_idx)
                elif data is not None:
                    data.append(node_result)
                if self.afterChild is not None:
                    self.afterChild(node, child, child_idx)
    finally:
        while ptr is not None:
            if hasattr(ptr[2], '__exit__'):
                ptr[2].__exit__(None, None, None)
            ptr = ptr[0]