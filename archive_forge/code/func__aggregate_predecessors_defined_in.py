import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
def _aggregate_predecessors_defined_in(self, node):
    preds = self.current_analyzer.graph.stmt_prev[node]
    node_defined_in = set()
    for p in preds:
        node_defined_in |= set(self.current_analyzer.out[p].value.keys())
    anno.setanno(node, anno.Static.DEFINED_VARS_IN, frozenset(node_defined_in))