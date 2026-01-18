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
def _create_nonlocal_declarations(self, vars_):
    vars_ = set(vars_)
    results = []
    global_vars = self.state[_Function].scope.globals & vars_
    if global_vars:
        results.append(gast.Global([str(v) for v in global_vars]))
    nonlocal_vars = [v for v in vars_ if not v.is_composite() and v not in global_vars]
    if nonlocal_vars:
        results.append(gast.Nonlocal([str(v) for v in nonlocal_vars]))
    return results