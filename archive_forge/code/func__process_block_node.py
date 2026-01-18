import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _process_block_node(self, node, block, scope_name):
    self._enter_scope(False)
    block = self.visit_block(block)
    self._exit_and_record_scope(node, tag=scope_name)
    return node