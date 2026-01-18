import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _process_annotation(self, node):
    self._in_annotation = True
    node = self.visit(node)
    self._in_annotation = False
    return node