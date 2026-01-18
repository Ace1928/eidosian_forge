import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _softmax_cross_entropy_with_logits_transformer(parent, node, full_name, name, logs):
    """Wrap labels argument with stop_gradients."""

    def _wrap_label(parent, old_value):
        """Wrap labels with tf.stop_gradient."""
        already_stop_grad = isinstance(old_value, ast.Call) and isinstance(old_value.func, ast.Attribute) and (old_value.func.attr == 'stop_gradient') and isinstance(old_value.func.value, ast.Name) and (old_value.func.value.id == 'tf')
        if already_stop_grad:
            return False
        try:
            new_value = ast.Call(ast.Name(id='tf.stop_gradient', ctx=ast.Load()), [old_value], [])
        except TypeError:
            new_value = ast.Call(ast.Name(id='tf.stop_gradient', ctx=ast.Load()), [old_value], [], None, None)
        pasta.ast_utils.replace_child(parent, old_value, new_value)
        ast.copy_location(new_value, old_value)
        return True
    for karg in node.keywords:
        if karg.arg == 'labels':
            if _wrap_label(karg, karg.value):
                logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing labels arg of tf.nn.softmax_cross_entropy_with_logits to tf.stop_gradient(labels). Please check this transformation.\n'))
            return node
    return node