import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _add_loss_reduction_transformer(parent, node, full_name, name, logs):
    """Adds a loss_reduction argument if not specified.

  Default value for tf.estimator.*Classifier and tf.estimator.*Regressor
  loss_reduction argument changed to SUM_OVER_BATCH_SIZE. So, we update
  existing calls to use the old default value `tf.keras.losses.Reduction.SUM`.

  Note: to apply this transformation, symbol must be added
  to reordered_function_names above.
  """
    for keyword_arg in node.keywords:
        if keyword_arg.arg == 'loss_reduction':
            return node
    default_value = 'tf.keras.losses.Reduction.SUM'
    ast_value = pasta.parse(default_value)
    node.keywords.append(ast.keyword(arg='loss_reduction', value=ast_value))
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, '%s: Default value of loss_reduction has been changed to SUM_OVER_BATCH_SIZE; inserting old default value %s.\n' % (full_name or name, default_value)))
    return node