from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import six
from pasta.augment import import_utils
from pasta.base import ast_utils
from pasta.base import scope
def _rename_reads(sc, t, old_name, new_name):
    """Updates all locations in the module where the given name is read.

  Arguments:
    sc: (scope.Scope) Scope to work in. This should be the scope of `t`.
    t: (ast.AST) The AST to perform updates in.
    old_name: (string) Dotted name to update.
    new_name: (string) Dotted name to replace it with.

  Returns:
    True if any changes were made, False otherwise.
  """
    name_parts = old_name.split('.')
    try:
        name = sc.names[name_parts[0]]
        for part in name_parts[1:]:
            name = name.attrs[part]
    except KeyError:
        return False
    has_changed = False
    for ref_node in name.reads:
        if isinstance(ref_node, (ast.Name, ast.Attribute)):
            ast_utils.replace_child(sc.parent(ref_node), ref_node, ast.parse(new_name).body[0].value)
            has_changed = True
    return has_changed