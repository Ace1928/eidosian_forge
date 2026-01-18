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
def _create_loop_options(self, node):
    if not anno.hasanno(node, anno.Basic.DIRECTIVES):
        return gast.Dict([], [])
    loop_directives = anno.getanno(node, anno.Basic.DIRECTIVES)
    if directives.set_loop_options not in loop_directives:
        return gast.Dict([], [])
    opts_dict = loop_directives[directives.set_loop_options]
    str_keys, values = zip(*opts_dict.items())
    keys = [gast.Constant(s, kind=None) for s in str_keys]
    values = list(values)
    return gast.Dict(keys, values)