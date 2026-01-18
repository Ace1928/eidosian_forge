import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _process_statement(self, node):
    self._enter_scope(False)
    node = self.generic_visit(node)
    self._exit_and_record_scope(node)
    return node