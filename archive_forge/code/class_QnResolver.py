import collections
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
class QnResolver(gast.NodeTransformer):
    """Annotates nodes with QN information.

  Note: Not using NodeAnnos to avoid circular dependencies.
  """

    def visit_Name(self, node):
        node = self.generic_visit(node)
        anno.setanno(node, anno.Basic.QN, QN(node.id))
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if anno.hasanno(node.value, anno.Basic.QN):
            anno.setanno(node, anno.Basic.QN, QN(anno.getanno(node.value, anno.Basic.QN), attr=node.attr))
        return node

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        s = node.slice
        if isinstance(s, (gast.Tuple, gast.Slice)):
            return node
        if isinstance(s, gast.Constant) and s.value != Ellipsis:
            subscript = QN(Literal(s.value))
        elif anno.hasanno(s, anno.Basic.QN):
            subscript = anno.getanno(s, anno.Basic.QN)
        else:
            return node
        if anno.hasanno(node.value, anno.Basic.QN):
            anno.setanno(node, anno.Basic.QN, QN(anno.getanno(node.value, anno.Basic.QN), subscript=subscript))
        return node