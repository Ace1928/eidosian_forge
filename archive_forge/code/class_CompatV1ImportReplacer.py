import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
class CompatV1ImportReplacer(ast.NodeVisitor):
    """AST Visitor that replaces `import tensorflow.compat.v1 as tf`.

  Converts `import tensorflow.compat.v1 as tf` to `import tensorflow as tf`
  """

    def visit_Import(self, node):
        """Handle visiting an import node in the AST.

    Args:
      node: Current Node
    """
        for import_alias in node.names:
            if import_alias.name == 'tensorflow.compat.v1' and import_alias.asname == 'tf':
                import_alias.name = 'tensorflow'
        self.generic_visit(node)