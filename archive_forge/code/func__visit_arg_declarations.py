import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _visit_arg_declarations(self, node):
    node.args.posonlyargs = self._visit_node_list(node.args.posonlyargs)
    node.args.args = self._visit_node_list(node.args.args)
    if node.args.vararg is not None:
        node.args.vararg = self.visit(node.args.vararg)
    node.args.kwonlyargs = self._visit_node_list(node.args.kwonlyargs)
    if node.args.kwarg is not None:
        node.args.kwarg = self.visit(node.args.kwarg)
    return node