import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
class ListTransformer(converter.Base):
    """Converts lists and related operations to their TF counterpart."""

    def visit_List(self, node):
        node = self.generic_visit(node)
        template = '\n      ag__.new_list(elements)\n    '
        return templates.replace_as_expression(template, elements=node)

    def _replace_append_call(self, node):
        assert len(node.args) == 1
        assert isinstance(node.func, gast.Attribute)
        template = '\n      target = ag__.list_append(target, element)\n    '
        return templates.replace(template, target=node.func.value, element=node.args[0])

    def _replace_pop_call(self, node):
        assert isinstance(node.func, gast.Attribute)
        scope = anno.getanno(node, NodeAnno.ARGS_SCOPE)
        target_node = node.func.value
        if anno.hasanno(target_node, anno.Basic.QN):
            target_name = anno.getanno(target_node, anno.Basic.QN).ssf()
        else:
            target_name = 'list_'
        pop_var_name = self.ctx.namer.new_symbol(target_name, scope.referenced)
        stmt = self.state[_Statement]
        if stmt.pop_uses is None:
            stmt.pop_uses = []
        stmt.pop_uses.append((node, pop_var_name))
        return templates.replace_as_expression('var_name', var_name=pop_var_name)

    def _replace_stack_call(self, node):
        assert len(node.args) == 1
        dtype = self.get_definition_directive(node.args[0], directives.set_element_type, 'dtype', default=templates.replace_as_expression('None'))
        template = '\n      ag__.list_stack(\n          target,\n          opts=ag__.ListStackOpts(\n              element_dtype=dtype,\n              original_call=orig_call))\n    '
        return templates.replace_as_expression(template, dtype=dtype, target=node.args[0], orig_call=node.func)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, gast.Attribute):
            func_name = node.func.attr
            if func_name == 'append' and len(node.args) == 1:
                node = self._replace_append_call(node)
            elif func_name == 'pop' and len(node.args) <= 1:
                node = self._replace_pop_call(node)
            elif func_name == 'stack' and len(node.args) == 1 and (not node.keywords or node.keywords[0].arg == 'strict'):
                node = self._replace_stack_call(node)
        return node

    def _generate_pop_operation(self, original_call_node, pop_var_name):
        assert isinstance(original_call_node.func, gast.Attribute)
        if original_call_node.args:
            pop_element = original_call_node.args[0]
        else:
            pop_element = parser.parse_expression('None')
        dtype = self.get_definition_directive(original_call_node.func.value, directives.set_element_type, 'dtype', default=templates.replace_as_expression('None'))
        shape = self.get_definition_directive(original_call_node.func.value, directives.set_element_type, 'shape', default=templates.replace_as_expression('None'))
        template = '\n      target, pop_var_name = ag__.list_pop(\n          target, element,\n          opts=ag__.ListPopOpts(element_dtype=dtype, element_shape=shape))\n    '
        return templates.replace(template, target=original_call_node.func.value, pop_var_name=pop_var_name, element=pop_element, dtype=dtype, shape=shape)

    def _postprocess_statement(self, node):
        """Inserts any separate pop() calls that node may use."""
        pop_uses = self.state[_Statement].pop_uses
        if pop_uses:
            replacements = []
            for original_call_node, pop_var_name in pop_uses:
                replacements.extend(self._generate_pop_operation(original_call_node, pop_var_name))
            replacements.append(node)
            node = replacements
        self.state[_Statement].exit()
        return (node, None)

    def _visit_and_process_block(self, block):
        return self.visit_block(block, before_visit=self.state[_Statement].enter, after_visit=self._postprocess_statement)

    def visit_FunctionDef(self, node):
        node.args = self.generic_visit(node.args)
        node.decorator_list = self.visit_block(node.decorator_list)
        node.body = self._visit_and_process_block(node.body)
        return node

    def visit_For(self, node):
        node.target = self.visit(node.target)
        node.body = self._visit_and_process_block(node.body)
        node.orelse = self._visit_and_process_block(node.orelse)
        return node

    def visit_While(self, node):
        node.test = self.visit(node.test)
        node.body = self._visit_and_process_block(node.body)
        node.orelse = self._visit_and_process_block(node.orelse)
        return node

    def visit_If(self, node):
        node.test = self.visit(node.test)
        node.body = self._visit_and_process_block(node.body)
        node.orelse = self._visit_and_process_block(node.orelse)
        return node

    def visit_With(self, node):
        node.items = self.visit_block(node.items)
        node.body = self._visit_and_process_block(node.body)
        return node