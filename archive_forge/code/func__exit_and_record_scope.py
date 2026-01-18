import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _exit_and_record_scope(self, node, tag=anno.Static.SCOPE):
    node_scope = self._exit_scope()
    anno.setanno(node, tag, node_scope)
    return node_scope