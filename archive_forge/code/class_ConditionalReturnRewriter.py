import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
class ConditionalReturnRewriter(converter.Base):
    """Rewrites a pattern where it's unobvious that all paths return a value.

  This rewrite allows avoiding intermediate None return values.

  The following pattern:

      if cond:
        <block 1>
        return
      else:
        <block 2>
      <block 3>

  is converted to:

      if cond:
        <block 1>
        return
      else:
        <block 2>
        <block 3>

  and vice-versa (if the else returns, subsequent statements are moved under the
  if branch).
  """

    def visit_Return(self, node):
        self.state[_RewriteBlock].definitely_returns = True
        return node

    def _postprocess_statement(self, node):
        if anno.getanno(node, STMT_DEFINITELY_RETURNS, default=False):
            self.state[_RewriteBlock].definitely_returns = True
        if isinstance(node, gast.If) and anno.getanno(node, BODY_DEFINITELY_RETURNS, default=False):
            return (node, node.orelse)
        elif isinstance(node, gast.If) and anno.getanno(node, ORELSE_DEFINITELY_RETURNS, default=False):
            return (node, node.body)
        return (node, None)

    def _visit_statement_block(self, node, nodes):
        self.state[_RewriteBlock].enter()
        new_nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
        block_definitely_returns = self.state[_RewriteBlock].definitely_returns
        self.state[_RewriteBlock].exit()
        return (new_nodes, block_definitely_returns)

    def visit_While(self, node):
        node.test = self.visit(node.test)
        node.body, _ = self._visit_statement_block(node, node.body)
        node.orelse, _ = self._visit_statement_block(node, node.orelse)
        return node

    def visit_For(self, node):
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)
        node.body, _ = self._visit_statement_block(node, node.body)
        node.orelse, _ = self._visit_statement_block(node, node.orelse)
        return node

    def visit_With(self, node):
        node.items = self.visit_block(node.items)
        node.body, definitely_returns = self._visit_statement_block(node, node.body)
        if definitely_returns:
            anno.setanno(node, STMT_DEFINITELY_RETURNS, True)
        return node

    def visit_Try(self, node):
        node.body, _ = self._visit_statement_block(node, node.body)
        node.orelse, _ = self._visit_statement_block(node, node.orelse)
        node.finalbody, _ = self._visit_statement_block(node, node.finalbody)
        node.handlers = self.visit_block(node.handlers)
        return node

    def visit_ExceptHandler(self, node):
        node.body, _ = self._visit_statement_block(node, node.body)
        return node

    def visit_If(self, node):
        node.test = self.visit(node.test)
        node.body, body_definitely_returns = self._visit_statement_block(node, node.body)
        if body_definitely_returns:
            anno.setanno(node, BODY_DEFINITELY_RETURNS, True)
        node.orelse, orelse_definitely_returns = self._visit_statement_block(node, node.orelse)
        if orelse_definitely_returns:
            anno.setanno(node, ORELSE_DEFINITELY_RETURNS, True)
        if body_definitely_returns and orelse_definitely_returns:
            self.state[_RewriteBlock].definitely_returns = True
        return node

    def visit_FunctionDef(self, node):
        node.args = self.visit(node.args)
        node.body, _ = self._visit_statement_block(node, node.body)
        return node