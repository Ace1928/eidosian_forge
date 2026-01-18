import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _enter_scope(self, isolated, f_name=None):
    self.scope = Scope(self.scope, isolated=isolated, function_name=f_name)