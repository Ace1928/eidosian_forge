import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _process_function_def(self, node, is_lambda):
    if self.builder is not None:
        self.builder.add_ordinary_node(node)
    self.builder_stack.append(self.builder)
    self.builder = GraphBuilder(node)
    self._enter_lexical_scope(node)
    self.builder.enter_section(node)
    self._process_basic_statement(node.args)
    if is_lambda:
        self._process_exit_statement(node.body, (gast.Lambda,))
    else:
        for stmt in node.body:
            self.visit(stmt)
    self.builder.exit_section(node)
    self._exit_lexical_scope(node)
    self.cfgs[node] = self.builder.build()
    self.builder = self.builder_stack.pop()