import ast
import inspect
import textwrap
import warnings
import torch
def _is_empty_container(self, node: ast.AST, ann_type: str) -> bool:
    if ann_type == 'List':
        if not isinstance(node, ast.List):
            return False
        if node.elts:
            return False
    elif ann_type == 'Dict':
        if not isinstance(node, ast.Dict):
            return False
        if node.keys:
            return False
    elif ann_type == 'Optional':
        if not isinstance(node, ast.Constant):
            return False
        if node.value:
            return False
    return True