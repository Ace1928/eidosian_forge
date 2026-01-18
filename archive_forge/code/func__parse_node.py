import ast
import sys
from typing import Any
from .internal import Filters, Key
def _parse_node(node) -> Filters:
    if isinstance(node, ast.Compare):
        if isinstance(node.left, ast.Call):
            func_call_data = _handle_function_call(node.left)
            if func_call_data:
                section = section_map.get(func_call_data['type'], 'default_section')
                key = Key(section=section, name=func_call_data['value'])
                op = _map_op(node.ops[0])
                right_operand = _extract_value(node.comparators[0])
                return Filters(op=op, key=key, value=right_operand, disabled=False)
        else:
            return _handle_comparison(node)
    elif isinstance(node, ast.BoolOp):
        return _handle_logical_op(node)
    else:
        raise ValueError(f'Unsupported expression type: {type(node)}')