import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _node_sets_self_attribute(self, node):
    if anno.hasanno(node, anno.Basic.QN):
        qn = anno.getanno(node, anno.Basic.QN)
        if qn.has_attr and qn.parent.qn == ('self',):
            return True
    return False