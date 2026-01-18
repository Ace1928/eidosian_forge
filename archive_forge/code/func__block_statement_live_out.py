import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import annos
def _block_statement_live_out(self, node):
    successors = self.current_analyzer.graph.stmt_next[node]
    stmt_live_out = set()
    for s in successors:
        stmt_live_out.update(self.current_analyzer.in_[s])
    anno.setanno(node, anno.Static.LIVE_VARS_OUT, frozenset(stmt_live_out))
    return node