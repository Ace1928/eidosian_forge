import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _rename_if_arg_found_and_add_loss_reduction_transformer(parent, node, full_name, name, logs, arg_names=None, arg_ok_predicate=None, remove_if_ok=False, message=None):
    """Combination of _rename_if_arg_found and _add_loss_reduction transformers.

  Args:
    parent: Parent of node.
    node: ast.Call node to maybe modify.
    full_name: full name of function to modify
    name: name of function to modify
    logs: list of logs to append to
    arg_names: list of names of the argument to look for
    arg_ok_predicate: predicate callable with the ast of the argument value,
      returns whether the argument value is allowed.
    remove_if_ok: remove the argument if present and ok as determined by
      arg_ok_predicate.
    message: message to print if a non-ok arg is found (and hence, the function
      is renamed to its compat.v1 version).

  Returns:
    node, if it was modified, else None.
  """
    node = _add_loss_reduction_transformer(parent, node, full_name, name, logs)
    for arg_name in arg_names:
        rename_node = _rename_if_arg_found_transformer(parent, node, full_name, name, logs, arg_name, arg_ok_predicate, remove_if_ok, message)
        node = rename_node if rename_node else node
    return node