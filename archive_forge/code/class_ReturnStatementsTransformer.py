import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
class ReturnStatementsTransformer(converter.Base):
    """Lowers return statements into variables and conditionals.

  Specifically, the following pattern:

      <block 1>
      return val
      <block 2>

  is converted to:

      do_return = False
      retval = None

      <block 1>

      do_return = True
      retval = val

      if not do_return:
        <block 2>

      return retval

  The conversion adjusts loops as well:

      <block 1>
      while cond:
        <block 2>
        return retval

  is converted to:

      <block 1>
      while not do_return and cond:
        <block 2>
        do_return = True
        retval = val
  """

    def __init__(self, ctx, allow_missing_return):
        super(ReturnStatementsTransformer, self).__init__(ctx)
        self.allow_missing_return = allow_missing_return

    def visit_Return(self, node):
        for block in reversed(self.state[_Block].stack):
            block.return_used = True
            block.create_guard_next = True
            if block.is_function:
                break
        retval = node.value if node.value else parser.parse_expression('None')
        template = '\n      try:\n        do_return_var_name = True\n        retval_var_name = retval\n      except:\n        do_return_var_name = False\n        raise\n    '
        node = templates.replace(template, do_return_var_name=self.state[_Function].do_return_var_name, retval_var_name=self.state[_Function].retval_var_name, retval=retval)
        return node

    def _postprocess_statement(self, node):
        if not self.state[_Block].return_used:
            return (node, None)
        state = self.state[_Block]
        if state.create_guard_now:
            template = '\n        if not do_return_var_name:\n          original_node\n      '
            cond, = templates.replace(template, do_return_var_name=self.state[_Function].do_return_var_name, original_node=node)
            node, block = (cond, cond.body)
        else:
            node, block = (node, None)
        state.create_guard_now = state.create_guard_next
        state.create_guard_next = False
        return (node, block)

    def _visit_statement_block(self, node, nodes):
        self.state[_Block].enter()
        nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
        self.state[_Block].exit()
        return nodes

    def visit_While(self, node):
        node.test = self.visit(node.test)
        node.body = self._visit_statement_block(node, node.body)
        if self.state[_Block].return_used:
            node.test = templates.replace_as_expression('not control_var and test', test=node.test, control_var=self.state[_Function].do_return_var_name)
        node.orelse = self._visit_statement_block(node, node.orelse)
        return node

    def visit_For(self, node):
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)
        node.body = self._visit_statement_block(node, node.body)
        if self.state[_Block].return_used:
            extra_test = anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST, default=None)
            if extra_test is not None:
                extra_test = templates.replace_as_expression('not control_var and extra_test', extra_test=extra_test, control_var=self.state[_Function].do_return_var_name)
            else:
                extra_test = templates.replace_as_expression('not control_var', control_var=self.state[_Function].do_return_var_name)
            anno.setanno(node, anno.Basic.EXTRA_LOOP_TEST, extra_test)
        node.orelse = self._visit_statement_block(node, node.orelse)
        return node

    def visit_With(self, node):
        node.items = self.visit_block(node.items)
        node.body = self._visit_statement_block(node, node.body)
        return node

    def visit_Try(self, node):
        node.body = self._visit_statement_block(node, node.body)
        node.orelse = self._visit_statement_block(node, node.orelse)
        node.finalbody = self._visit_statement_block(node, node.finalbody)
        node.handlers = self.visit_block(node.handlers)
        return node

    def visit_ExceptHandler(self, node):
        node.body = self._visit_statement_block(node, node.body)
        return node

    def visit_If(self, node):
        node.test = self.visit(node.test)
        node.body = self._visit_statement_block(node, node.body)
        node.orelse = self._visit_statement_block(node, node.orelse)
        return node

    def visit_FunctionDef(self, node):
        with self.state[_Function] as fn:
            with self.state[_Block] as block:
                block.is_function = True
                scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
                do_return_var_name = self.ctx.namer.new_symbol('do_return', scope.referenced)
                retval_var_name = self.ctx.namer.new_symbol('retval_', scope.referenced)
                fn.do_return_var_name = do_return_var_name
                fn.retval_var_name = retval_var_name
                node.body = self._visit_statement_block(node, node.body)
                if block.return_used:
                    if self.allow_missing_return:
                        wrapper_node = node.body[-1]
                        assert isinstance(wrapper_node, gast.With), 'This transformer requires the functions converter.'
                        template = '\n              do_return_var_name = False\n              retval_var_name = ag__.UndefinedReturnValue()\n              body\n              return function_context.ret(retval_var_name, do_return_var_name)\n            '
                        wrapper_node.body = templates.replace(template, body=wrapper_node.body, do_return_var_name=do_return_var_name, function_context=anno.getanno(node, 'function_context_name'), retval_var_name=retval_var_name)
                    else:
                        template = '\n              body\n              return retval_var_name\n            '
                        node.body = templates.replace(template, body=node.body, do_return_var_name=do_return_var_name, retval_var_name=retval_var_name)
        return node