import ast
import sys
from typing import Any
from .internal import Filters, Key
def _map_op(op_node) -> str:
    op_map = {ast.Gt: '>', ast.Lt: '<', ast.Eq: '==', ast.NotEq: '!=', ast.GtE: '>=', ast.LtE: '<=', ast.In: 'IN', ast.NotIn: 'NIN'}
    return op_map[type(op_node)]