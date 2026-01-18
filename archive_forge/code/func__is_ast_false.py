import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _is_ast_false(node):
    if hasattr(ast, 'NameConstant'):
        return isinstance(node, ast.NameConstant) and node.value is False
    else:
        return isinstance(node, ast.Name) and node.id == 'False'