import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _contrib_layers_variance_scaling_initializer_transformer(parent, node, full_name, name, logs):
    """Updates references to contrib.layers.variance_scaling_initializer.

  Transforms:
  tf.contrib.layers.variance_scaling_initializer(
    factor, mode, uniform, seed, dtype
  ) to
  tf.compat.v1.keras.initializers.VarianceScaling(
      scale=factor, mode=mode.lower(),
      distribution=("uniform" if uniform else "truncated_normal"),
      seed=seed, dtype=dtype)

  And handles the case where no factor is provided and scale needs to be
  set to 2.0 to match contrib's default instead of tf.keras.initializer's
  default of 1.0
  """

    def _replace_distribution(parent, old_value):
        """Replaces old_value: ("uniform" if (old_value) else "truncated_normal")"""
        new_value = pasta.parse('"uniform" if old_value else "truncated_normal"')
        ifexpr = new_value.body[0].value
        pasta.ast_utils.replace_child(ifexpr, ifexpr.test, old_value)
        pasta.ast_utils.replace_child(parent, old_value, new_value)
        pasta.base.formatting.set(new_value, 'prefix', '(')
        pasta.base.formatting.set(new_value, 'suffix', ')')

    def _replace_mode(parent, old_value):
        """Replaces old_value with (old_value).lower()."""
        new_value = pasta.parse('mode.lower()')
        mode = new_value.body[0].value.func
        pasta.ast_utils.replace_child(mode, mode.value, old_value)
        pasta.ast_utils.replace_child(parent, old_value, new_value)
        pasta.base.formatting.set(old_value, 'prefix', '(')
        pasta.base.formatting.set(old_value, 'suffix', ')')
    found_scale = False
    for keyword_arg in node.keywords:
        if keyword_arg.arg == 'factor':
            keyword_arg.arg = 'scale'
            found_scale = True
        if keyword_arg.arg == 'mode':
            _replace_mode(keyword_arg, keyword_arg.value)
        if keyword_arg.arg == 'uniform':
            keyword_arg.arg = 'distribution'
            _replace_distribution(keyword_arg, keyword_arg.value)
    if len(node.args) >= 1:
        found_scale = True
    if len(node.args) >= 2:
        _replace_mode(node, node.args[1])
    if len(node.args) >= 3:
        _replace_distribution(node, node.args[2])
    if not found_scale:
        scale_value = pasta.parse('2.0')
        node.keywords = [ast.keyword(arg='scale', value=scale_value)] + node.keywords
    lineno = node.func.value.lineno
    col_offset = node.func.value.col_offset
    node.func.value = ast_edits.full_name_node('tf.compat.v1.keras.initializers')
    node.func.value.lineno = lineno
    node.func.value.col_offset = col_offset
    node.func.attr = 'VarianceScaling'
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing tf.contrib.layers.variance_scaling_initializer to a tf.compat.v1.keras.initializers.VarianceScaling and converting arguments.\n'))
    return node