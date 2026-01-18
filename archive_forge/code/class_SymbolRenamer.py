import ast
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
class SymbolRenamer(gast.NodeTransformer):
    """Transformer that can rename symbols to a simple names."""

    def __init__(self, name_map):
        self.name_map = name_map

    def _process_name_node(self, node):
        qn = anno.getanno(node, anno.Basic.QN)
        if qn in self.name_map:
            new_node = gast.Name(str(self.name_map[qn]), ctx=node.ctx, annotation=None, type_comment=None)
            for k in anno.keys(node):
                anno.copyanno(node, new_node, k)
            return new_node
        return self.generic_visit(node)

    def _process_list_of_strings(self, names):
        for i in range(len(names)):
            qn = qual_names.QN(names[i])
            if qn in self.name_map:
                names[i] = str(self.name_map[qn])
        return names

    def visit_Nonlocal(self, node):
        node.names = self._process_list_of_strings(node.names)
        return node

    def visit_Global(self, node):
        node.names = self._process_list_of_strings(node.names)
        return node

    def visit_Name(self, node):
        return self._process_name_node(node)

    def visit_Attribute(self, node):
        if anno.hasanno(node, anno.Basic.QN):
            return self._process_name_node(node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        qn = qual_names.QN(node.name)
        if qn in self.name_map:
            node.name = str(self.name_map[qn])
        return self.generic_visit(node)