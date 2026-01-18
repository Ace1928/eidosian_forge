import ast
import sys
from typing import Any
from .internal import Filters, Key
class CustomNodeVisitor(ast.NodeVisitor):

    def visit_Compare(self, node):
        left = self.handle_expression(node.left)
        print(f'Expression type: {left}')
        self.generic_visit(node)

    def handle_expression(self, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in ['Config', 'SummaryMetric', 'KeysInfo', 'Tags', 'Metric']:
                if len(node.args) == 1 and isinstance(node.args[0], ast.Str):
                    arg_value = node.args[0].s
                    return (func_name, arg_value)
        return self.get_full_expression(node)

    def get_full_expression(self, node):
        if isinstance(node, ast.Attribute):
            return self.get_full_expression(node.value) + '.' + node.attr
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return 'ArbitraryExpression'