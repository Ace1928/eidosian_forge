import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _add_uniform_scaling_initializer_transformer(parent, node, full_name, name, logs):
    """Updates references to uniform_unit_scaling_initializer.

  Transforms:
  tf.uniform_unit_scaling_initializer(factor, seed, dtype) to
  tf.compat.v1.keras.initializers.VarianceScaling(
      scale=factor, distribution="uniform", seed=seed)

  Note: to apply this transformation, symbol must be added
  to reordered_function_names above.
  """
    for keyword_arg in node.keywords:
        if keyword_arg.arg == 'factor':
            keyword_arg.arg = 'scale'
    distribution_value = '"uniform"'
    ast_value = pasta.parse(distribution_value)
    node.keywords.append(ast.keyword(arg='distribution', value=ast_value))
    lineno = node.func.value.lineno
    col_offset = node.func.value.col_offset
    node.func.value = ast_edits.full_name_node('tf.compat.v1.keras.initializers')
    node.func.value.lineno = lineno
    node.func.value.col_offset = col_offset
    node.func.attr = 'VarianceScaling'
    return node