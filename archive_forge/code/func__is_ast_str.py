import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _is_ast_str(node):
    """Determine whether this node represents a string."""
    allowed_types = [ast.Str]
    if hasattr(ast, 'Bytes'):
        allowed_types += [ast.Bytes]
    if hasattr(ast, 'JoinedStr'):
        allowed_types += [ast.JoinedStr]
    if hasattr(ast, 'FormattedValue'):
        allowed_types += [ast.FormattedValue]
    return isinstance(node, allowed_types)