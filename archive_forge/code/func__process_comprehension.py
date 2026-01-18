import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _process_comprehension(self, node, is_list_comp=False, is_dict_comp=False):
    with self.state[_Comprehension] as comprehension_:
        comprehension_.is_list_comp = is_list_comp
        node.generators = self.visit_block(node.generators)
        if is_dict_comp:
            node.key = self.visit(node.key)
            node.value = self.visit(node.value)
        else:
            node.elt = self.visit(node.elt)
        return node