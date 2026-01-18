import inspect
import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.util import tf_inspect
class DirectivesTransformer(converter.Base):
    """Parses compiler directives and converts them into AST annotations."""

    def _process_symbol_directive(self, call_node, directive):
        if len(call_node.args) < 1:
            raise ValueError('"%s" requires a positional first argument as the target' % directive.__name__)
        target = call_node.args[0]
        defs = anno.getanno(target, anno.Static.ORIG_DEFINITIONS)
        for def_ in defs:
            def_.directives[directive] = _map_args(call_node, directive)
        return call_node

    def _process_statement_directive(self, call_node, directive):
        if self.state[_LoopScope].statements_visited > 1:
            raise ValueError('"%s" must be the first statement in the loop block' % directive.__name__)
        if self.state[_LoopScope].level < 2:
            raise ValueError('"%s" must be used inside a statement' % directive.__name__)
        target = self.state[_LoopScope].ast_node
        node_anno = anno.getanno(target, anno.Basic.DIRECTIVES, {})
        node_anno[directive] = _map_args(call_node, directive)
        anno.setanno(target, anno.Basic.DIRECTIVES, node_anno)
        return call_node

    def visit_Name(self, node):
        node = self.generic_visit(node)
        if isinstance(node.ctx, gast.Load):
            defs = anno.getanno(node, anno.Static.DEFINITIONS, ())
            is_defined = bool(defs)
            if not is_defined and node.id in self.ctx.info.namespace:
                anno.setanno(node, STATIC_VALUE, self.ctx.info.namespace[node.id])
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        parent_val = anno.getanno(node.value, STATIC_VALUE, default=None)
        if parent_val is not None and inspect.ismodule(parent_val):
            if hasattr(parent_val, node.attr):
                anno.setanno(node, STATIC_VALUE, getattr(parent_val, node.attr))
        return node

    def visit_Assign(self, node):
        self.state[_LoopScope].statements_visited += 1
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.state[_LoopScope].statements_visited += 1
        return self.generic_visit(node)

    def visit_Expr(self, node):
        self.state[_LoopScope].statements_visited += 1
        node = self.generic_visit(node)
        if isinstance(node.value, gast.Call):
            call_node = node.value
            static_val = anno.getanno(call_node.func, STATIC_VALUE, default=None)
            if static_val is not None:
                if static_val is directives.set_element_type:
                    self._process_symbol_directive(call_node, static_val)
                    return None
                elif static_val is directives.set_loop_options:
                    self._process_statement_directive(call_node, static_val)
                    return None
        return node

    def _track_and_visit_loop(self, node):
        self.state[_LoopScope].enter()
        self.state[_LoopScope].ast_node = node
        node = self.generic_visit(node)
        if not node.body:
            node.body = [gast.Pass()]
        self.state[_LoopScope].exit()
        return node

    def visit_While(self, node):
        return self._track_and_visit_loop(node)

    def visit_For(self, node):
        return self._track_and_visit_loop(node)