import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
class TestStreamBasedExpressionVisitor_Deep(unittest.TestCase):

    def setUp(self):
        self.m = m = ConcreteModel()
        m.x = Var()
        m.I = Set(initialize=range(2 * RECURSION_LIMIT))

        def _rule(m, i):
            if i:
                return m.e[i - 1]
            else:
                return m.x
        m.e = Expression(m.I, rule=_rule)

    def evaluate_bx(self):

        def before(node, child, idx):
            if type(child) in native_types or not child.is_expression_type():
                return (False, value(child))
            return (True, None)

        def exit(node, data):
            return data[0] + 1
        return StreamBasedExpressionVisitor(beforeChild=before, exitNode=exit)

    def evaluate_bex(self):

        def before(node, child, idx):
            if type(child) in native_types or not child.is_expression_type():
                return (False, value(child))
            return (True, None)

        def enter(node):
            return (None, [])

        def exit(node, data):
            return data[0] + 1
        return StreamBasedExpressionVisitor(beforeChild=before, enterNode=enter, exitNode=exit)

    def evaluate_abex(self):

        def before(node, child, idx):
            if type(child) in native_types or not child.is_expression_type():
                return (False, value(child))
            return (True, None)

        def enter(node):
            return (None, 0)

        def accept(node, data, child_result, child_idx):
            return data + child_result

        def exit(node, data):
            return data + 1
        return StreamBasedExpressionVisitor(beforeChild=before, acceptChildResult=accept, enterNode=enter, exitNode=exit)

    def run_walker(self, walker):
        m = self.m
        m.x = 10
        self.assertEqual(2 * RECURSION_LIMIT + 10, walker.walk_expression(m.e[2 * RECURSION_LIMIT - 1]))
        self.assertEqual(2 * RECURSION_LIMIT + 10, walker.walk_expression_nonrecursive(m.e[2 * RECURSION_LIMIT - 1]))
        TESTING_OVERHEAD = 14
        warn_msg = 'Unexpected RecursionError walking an expression tree.\n'
        if platform.python_implementation() == 'PyPy':
            cases = [(0, '')]
        elif os.environ.get('GITHUB_ACTIONS', '') and sys.platform.startswith('win'):
            cases = []
        else:
            cases = [(0, ''), (10, warn_msg)]
        head_room = sys.getrecursionlimit() - get_stack_depth()
        for n, msg in cases:
            with LoggingIntercept() as LOG:
                self.assertEqual(2 * RECURSION_LIMIT + 10, fill_stack(head_room - RECURSION_LIMIT - TESTING_OVERHEAD + n, walker.walk_expression, m.e[2 * RECURSION_LIMIT - 1]))
            self.assertEqual(msg, LOG.getvalue())

    def test_evaluate_bx(self):
        return self.run_walker(self.evaluate_bx())

    def test_evaluate_bex(self):
        return self.run_walker(self.evaluate_bex())

    def test_evaluate_abex(self):
        return self.run_walker(self.evaluate_abex())