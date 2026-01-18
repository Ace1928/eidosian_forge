import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def _is_trivial(node):
    """Returns whether to consider the given node 'trivial'.

  The definition of 'trivial' is a node that can't meaningfully be pulled out
  into its own assignment statement.

  This is surprisingly difficult to do robustly across versions of Python and
  gast, as the parsing of constants has changed, if I may, constantly.

  Args:
    node: An AST node to check for triviality

  Returns:
    trivial: A Python `bool` indicating whether the node is trivial.
  """
    trivial_node_types = (gast.Name, bool, str, gast.Add, gast.Sub, gast.Mult, gast.Div, gast.Mod, gast.Pow, gast.LShift, gast.RShift, gast.BitOr, gast.BitXor, gast.BitAnd, gast.FloorDiv, gast.Invert, gast.Not, gast.UAdd, gast.USub, gast.Eq, gast.NotEq, gast.Lt, gast.LtE, gast.Gt, gast.GtE, gast.Is, gast.IsNot, gast.In, gast.NotIn, gast.expr_context)
    if isinstance(node, trivial_node_types) and (not _is_py2_name_constant(node)):
        return True
    if gast_util.is_ellipsis(node):
        return True
    return False