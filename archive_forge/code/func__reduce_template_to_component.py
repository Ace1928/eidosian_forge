import itertools
import logging
import sys
import builtins
from contextlib import nullcontext
from pyomo.common.errors import TemplateExpressionError
from pyomo.core.expr.base import ExpressionBase, ExpressionArgs_Mixin, NPV_Mixin
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.relational_expr import tuple_to_relational_expr
from pyomo.core.expr.visitor import (
def _reduce_template_to_component(expr):
    """Resolve a template into a concrete component

    This takes a template expression and returns the concrete equivalent
    by substituting the current values of all IndexTemplate objects and
    resolving (evaluating and removing) all GetItemExpression,
    GetAttrExpression, and TemplateSumExpression expression nodes.

    """
    import pyomo.core.base.set
    wildcards = []
    wildcard_groups = {}
    level = -1

    def beforeChild(node, child, child_idx):
        if type(child) in native_types:
            return (False, child)
        elif not child.is_expression_type():
            if hasattr(child, '_resolve_template'):
                try:
                    ans = child._resolve_template(())
                except TemplateExpressionError:
                    if child._group not in wildcard_groups:
                        wildcard_groups[child._group] = len(wildcards)
                        info = _wildcard_info(child._set, child)
                        wildcards.append(info)
                    else:
                        info = wildcards[wildcard_groups[child._group]]
                        info.objects.append(child)
                        child.set_value(info.value)
                    ans = child._resolve_template(())
                return (False, ans)
            if child.is_variable_type():
                from pyomo.core.base.set import RangeSet
                if child.domain.isdiscrete():
                    domain = child.domain
                    bounds = child.bounds
                    if bounds != (None, None):
                        try:
                            bounds = pyomo.core.base.set.RangeSet(*bounds, 0)
                            domain = domain & bounds
                        except:
                            pass
                    info = _wildcard_info(domain, child)
                    wildcards.append(info)
                return (False, value(child))
            return (False, child)
        else:
            return (True, None)

    def exitNode(node, args):
        if hasattr(node, '_resolve_template'):
            return node._resolve_template(args)
        if len(args) == node.nargs() and all((a is b for a, b in zip(node.args, args))):
            return node
        if all(map(is_constant, args)):
            return node._apply_operation(args)
        else:
            return node.create_node_with_local_data(args)
    walker = StreamBasedExpressionVisitor(initializeWalker=lambda x: beforeChild(None, x, None), beforeChild=beforeChild, exitNode=exitNode)
    while 1:
        try:
            with _TemplateIterManager.pause():
                ans = walker.walk_expression(expr)
            break
        except (KeyError, AttributeError):
            level = len(wildcards) - 1
            while level >= 0:
                info = wildcards[level]
                try:
                    info.advance()
                    break
                except StopIteration:
                    info.reset()
                    info.advance()
                    level -= 1
            if level < 0:
                for info in wildcards:
                    info.restore()
                raise
    for info in wildcards:
        info.restore()
    return ans