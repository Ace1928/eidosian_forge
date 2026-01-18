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
def _create_state_functions(self, block_vars, nonlocal_declarations, getter_name, setter_name):
    if not block_vars:
        template = '\n        def getter_name():\n          return ()\n        def setter_name(block_vars):\n          pass\n      '
        return templates.replace(template, getter_name=getter_name, setter_name=setter_name)
    guarded_block_vars = []
    for v in block_vars:
        if v.is_simple():
            guarded_block_vars.append(v)
        else:
            guarded_block_vars.append(templates.replace_as_expression('ag__.ldu(lambda: var_, name)', var_=v, name=gast.Constant(str(v), kind=None)))
    template = '\n      def getter_name():\n        return guarded_state_vars,\n      def setter_name(vars_):\n        nonlocal_declarations\n        state_vars, = vars_\n    '
    return templates.replace(template, nonlocal_declarations=nonlocal_declarations, getter_name=getter_name, guarded_state_vars=guarded_block_vars, setter_name=setter_name, state_vars=tuple(block_vars))