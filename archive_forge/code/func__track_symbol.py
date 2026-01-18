import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _track_symbol(self, node, composite_writes_alter_parent=False):
    if self._track_annotations_only and (not self._in_annotation):
        return
    if not anno.hasanno(node, anno.Basic.QN):
        return
    qn = anno.getanno(node, anno.Basic.QN)
    for l in self.state[_Comprehension]:
        if qn in l.targets:
            return
        if qn.owner_set & set(l.targets):
            return
    if isinstance(node.ctx, gast.Store):
        if self.state[_Comprehension].level > 0:
            self.state[_Comprehension].targets.add(qn)
            return
        self.scope.modified.add(qn)
        self.scope.bound.add(qn)
        if qn.is_composite and composite_writes_alter_parent:
            self.scope.modified.add(qn.parent)
        if self._in_aug_assign:
            self.scope.read.add(qn)
    elif isinstance(node.ctx, gast.Load):
        self.scope.read.add(qn)
        if self._in_annotation:
            self.scope.annotations.add(qn)
    elif isinstance(node.ctx, gast.Param):
        self.scope.bound.add(qn)
        self.scope.mark_param(qn, self.state[_FunctionOrClass].node)
    elif isinstance(node.ctx, gast.Del):
        self.scope.read.add(qn)
        self.scope.bound.add(qn)
        self.scope.deleted.add(qn)
    else:
        raise ValueError('Unknown context {} for node "{}".'.format(type(node.ctx), qn))