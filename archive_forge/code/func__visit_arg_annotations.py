import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _visit_arg_annotations(self, node):
    node.args.kw_defaults = self._visit_node_list(node.args.kw_defaults)
    node.args.defaults = self._visit_node_list(node.args.defaults)
    self._track_annotations_only = True
    node = self._visit_arg_declarations(node)
    self._track_annotations_only = False
    return node