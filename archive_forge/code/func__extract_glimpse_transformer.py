import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _extract_glimpse_transformer(parent, node, full_name, name, logs):

    def _replace_uniform_noise_node(parent, old_value):
        """Replaces old_value with 'uniform' or 'gaussian'."""
        uniform = ast.Str(s='uniform')
        gaussian = ast.Str(s='gaussian')
        new_value = ast.IfExp(body=uniform, test=old_value, orelse=gaussian)
        pasta.ast_utils.replace_child(parent, old_value, new_value)
        ast.copy_location(new_value, old_value)
        pasta.base.formatting.set(new_value.test, 'prefix', '(')
        pasta.base.formatting.set(new_value.test, 'suffix', ')')
    for uniform_noise in node.keywords:
        if uniform_noise.arg == 'uniform_noise':
            logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing uniform_noise arg of tf.image.extract_glimpse to noise, and recomputing value. Please check this transformation.\n'))
            uniform_noise.arg = 'noise'
            value = 'uniform' if uniform_noise.value else 'gaussian'
            _replace_uniform_noise_node(uniform_noise, uniform_noise.value)
            return node
    if len(node.args) >= 5:
        _replace_uniform_noise_node(node, node.args[5])
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing uniform_noise arg of tf.image.extract_glimpse to noise, and recomputing value.\n'))
        return node