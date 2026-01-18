import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.autograph.pyct.static_analysis import liveness
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
def _get_block_basic_vars(self, modified, live_in, live_out):
    nonlocals = self.state[_Function].scope.nonlocals
    basic_scope_vars = []
    for s in modified:
        if s.is_composite():
            continue
        if s in live_in or s in live_out or s in nonlocals:
            basic_scope_vars.append(s)
        continue
    return frozenset(basic_scope_vars)