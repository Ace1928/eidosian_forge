from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _visit_loop_body(self, node, nodes):
    self.state[_Continue].enter()
    self.state[_Block].enter()
    self.state[_Block].is_loop_type = True
    scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    continue_var = self.ctx.namer.new_symbol('continue_', scope.referenced)
    self.state[_Continue].control_var_name = continue_var
    nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
    if self.state[_Continue].used:
        template = '\n        var_name = False\n      '
        control_var_init = templates.replace(template, var_name=continue_var)
        nodes = control_var_init + nodes
    self.state[_Block].exit()
    self.state[_Continue].exit()
    return nodes