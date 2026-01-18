import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
Puts `node` in A-normal form, by replacing it with a variable if needed.

    The exact definition of A-normal form is given by the configuration.  The
    parent and the incoming field name are only needed because the configuration
    may be context-dependent.

    Args:
      parent: An AST node, the parent of `node`.
      field: The field name under which `node` is the child of `parent`.
      node: An AST node, potentially to be replaced with a variable reference.

    Returns:
      node: An AST node; the argument if transformation was not necessary,
        or the new variable reference if it was.
    