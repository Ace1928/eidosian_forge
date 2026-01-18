import inspect
import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.util import tf_inspect
def _track_and_visit_loop(self, node):
    self.state[_LoopScope].enter()
    self.state[_LoopScope].ast_node = node
    node = self.generic_visit(node)
    if not node.body:
        node.body = [gast.Pass()]
    self.state[_LoopScope].exit()
    return node