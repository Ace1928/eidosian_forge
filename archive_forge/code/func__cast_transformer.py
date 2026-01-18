import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _cast_transformer(parent, node, full_name, name, logs):
    """Transforms to_int and to_float to cast(..., dtype=...)."""
    dtype_str = name[3:]
    if dtype_str == 'float':
        dtype_str = 'float32'
    elif dtype_str == 'double':
        dtype_str = 'float64'
    new_arg = ast.keyword(arg='dtype', value=ast.Attribute(value=ast.Name(id='tf', ctx=ast.Load()), attr=dtype_str, ctx=ast.Load()))
    if len(node.args) == 2:
        name_arg = ast.keyword(arg='name', value=node.args[-1])
        node.args = node.args[:-1]
        node.keywords.append(name_arg)
    new_arg.value.lineno = node.lineno
    new_arg.value.col_offset = node.col_offset + 100
    node.keywords.append(new_arg)
    if isinstance(node.func, ast.Attribute):
        node.func.attr = 'cast'
    else:
        assert isinstance(node.func, ast.Name)
        node.func.id = 'cast'
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changed %s call to tf.cast(..., dtype=tf.%s).' % (full_name, dtype_str)))
    return node