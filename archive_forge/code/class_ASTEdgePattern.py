import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
class ASTEdgePattern(collections.namedtuple('ASTEdgePattern', ['parent', 'field', 'child'])):
    """A pattern defining a type of AST edge.

  This consists of three components:
  - The type of the parent node, checked with isinstance,
  - The name of the field, checked with string equality, and
  - The type of the child node, also checked with isinstance.
  If all three match, the whole pattern is considered to match.

  In all three slots, the special value `anf.ANY` is treated as "match
  anything".  The internal nodes are produced from the `gast` library rather
  than the standard `ast` module, which may affect `isinstance` checks.
  """
    __slots__ = ()

    def matches(self, parent, field, child):
        """Computes whether this pattern matches the given edge."""
        if self.parent is ANY or isinstance(parent, self.parent):
            pass
        else:
            return False
        if self.field is ANY or field == self.field:
            pass
        else:
            return False
        return self.child is ANY or isinstance(child, self.child)