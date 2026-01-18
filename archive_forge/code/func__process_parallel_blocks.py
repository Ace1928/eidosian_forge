import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _process_parallel_blocks(self, parent, children):
    before_parent = Scope.copy_of(self.scope)
    after_children = []
    for child, scope_name in children:
        self.scope.copy_from(before_parent)
        parent = self._process_block_node(parent, child, scope_name)
        after_child = Scope.copy_of(self.scope)
        after_children.append(after_child)
    for after_child in after_children:
        self.scope.merge_from(after_child)
    return parent