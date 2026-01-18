import ast
import sys
from typing import Any
from .internal import Filters, Key
def _handle_function_call(node) -> dict:
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name in ['Config', 'SummaryMetric', 'KeysInfo', 'Tags', 'Metric']:
            if len(node.args) == 1 and isinstance(node.args[0], ast.Str):
                arg_value = node.args[0].s
                return {'type': func_name, 'value': arg_value}
            else:
                raise ValueError(f'Invalid arguments for {func_name}')
    else:
        raise ValueError('Unsupported function call')