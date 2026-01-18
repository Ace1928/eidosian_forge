import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _contrib_layers_xavier_initializer_transformer(parent, node, full_name, name, logs):
    """Updates references to contrib.layers.xavier_initializer.

  Transforms:
  tf.contrib.layers.xavier_initializer(uniform, seed, dtype) to
  tf.compat.v1.keras.initializers.VarianceScaling(
      scale=1.0, mode="fan_avg",
      distribution=("uniform" if uniform else "truncated_normal"),
      seed=seed, dtype=dtype)

  Returns: The new node
  """

    def _get_distribution(old_value):
        """Returns an AST matching the following:
    ("uniform" if (old_value) else "truncated_normal")
    """
        dist = pasta.parse('"uniform" if old_value else "truncated_normal"')
        ifexpr = dist.body[0].value
        pasta.ast_utils.replace_child(ifexpr, ifexpr.test, old_value)
        pasta.base.formatting.set(dist, 'prefix', '(')
        pasta.base.formatting.set(dist, 'suffix', ')')
        return dist
    found_distribution = False
    for keyword_arg in node.keywords:
        if keyword_arg.arg == 'uniform':
            found_distribution = True
            keyword_arg.arg = 'distribution'
            old_value = keyword_arg.value
            new_value = _get_distribution(keyword_arg.value)
            pasta.ast_utils.replace_child(keyword_arg, old_value, new_value)
            pasta.base.formatting.set(keyword_arg.value, 'prefix', '(')
            pasta.base.formatting.set(keyword_arg.value, 'suffix', ')')
    new_keywords = []
    scale = pasta.parse('1.0')
    new_keywords.append(ast.keyword(arg='scale', value=scale))
    mode = pasta.parse('"fan_avg"')
    new_keywords.append(ast.keyword(arg='mode', value=mode))
    if len(node.args) >= 1:
        found_distribution = True
        dist = _get_distribution(node.args[0])
        new_keywords.append(ast.keyword(arg='distribution', value=dist))
    if not found_distribution:
        uniform_dist = pasta.parse('"uniform"')
        new_keywords.append(ast.keyword(arg='distribution', value=uniform_dist))
    if len(node.args) >= 2:
        new_keywords.append(ast.keyword(arg='seed', value=node.args[1]))
    if len(node.args) >= 3:
        new_keywords.append(ast.keyword(arg='dtype', value=node.args[2]))
    node.args = []
    node.keywords = new_keywords + node.keywords
    lineno = node.func.value.lineno
    col_offset = node.func.value.col_offset
    node.func.value = ast_edits.full_name_node('tf.compat.v1.keras.initializers')
    node.func.value.lineno = lineno
    node.func.value.col_offset = col_offset
    node.func.attr = 'VarianceScaling'
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing tf.contrib.layers xavier initializer to a tf.compat.v1.keras.initializers.VarianceScaling and converting arguments.\n'))
    return node