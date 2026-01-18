import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
@property
def _in_constructor(self):
    context = self.state[_FunctionOrClass]
    if context.level > 2:
        innermost = context.stack[-1].node
        parent = context.stack[-2].node
        return isinstance(parent, gast.ClassDef) and (isinstance(innermost, gast.FunctionDef) and innermost.name == '__init__')
    return False