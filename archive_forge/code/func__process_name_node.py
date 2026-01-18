import ast
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
def _process_name_node(self, node):
    qn = anno.getanno(node, anno.Basic.QN)
    if qn in self.name_map:
        new_node = gast.Name(str(self.name_map[qn]), ctx=node.ctx, annotation=None, type_comment=None)
        for k in anno.keys(node):
            anno.copyanno(node, new_node, k)
        return new_node
    return self.generic_visit(node)