import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
class AnfTransformer(transformer.Base):
    """Performs the conversion to A-normal form (ANF)."""

    def __init__(self, ctx, config):
        """Creates an ANF transformer.

    Args:
      ctx: transformer.Context
      config: Configuration
    """
        super(AnfTransformer, self).__init__(ctx)
        if config is None:
            if gast_util.GAST2:
                literal_node_types = (gast.Num, gast.Str, gast.Bytes, gast.NameConstant, gast.Name)
            elif gast_util.GAST3:
                literal_node_types = (gast.Constant, gast.Name)
            else:
                assert False
            self._overrides = [(ASTEdgePattern(ANY, ANY, literal_node_types), LEAVE), (ASTEdgePattern(ANY, ANY, gast.expr), REPLACE)]
        else:
            self._overrides = config
        self._gensym = DummyGensym()
        self._pending_statements = []

    def _consume_pending_statements(self):
        ans = self._pending_statements
        self._pending_statements = []
        return ans

    def _add_pending_statement(self, stmt):
        self._pending_statements.append(stmt)

    def _match(self, pattern, parent, field, child):
        if pattern is ANY:
            return True
        else:
            return pattern.matches(parent, field, child)

    def _should_transform(self, parent, field, child):
        for pat, result in self._overrides:
            if self._match(pat, parent, field, child):
                return result(parent, field, child)
        return False

    def _do_transform_node(self, node):
        temp_name = self._gensym.new_name()
        temp_assign = templates.replace('temp_name = expr', temp_name=temp_name, expr=node)[0]
        self._add_pending_statement(temp_assign)
        answer = templates.replace('temp_name', temp_name=temp_name)[0]
        return answer

    def _ensure_node_in_anf(self, parent, field, node):
        """Puts `node` in A-normal form, by replacing it with a variable if needed.

    The exact definition of A-normal form is given by the configuration.  The
    parent and the incoming field name are only needed because the configuration
    may be context-dependent.

    Args:
      parent: An AST node, the parent of `node`.
      field: The field name under which `node` is the child of `parent`.
      node: An AST node, potentially to be replaced with a variable reference.

    Returns:
      node: An AST node; the argument if transformation was not necessary,
        or the new variable reference if it was.
    """
        if node is None:
            return node
        if _is_trivial(node):
            return node
        if isinstance(node, list):
            return [self._ensure_node_in_anf(parent, field, n) for n in node]
        if isinstance(node, gast.keyword):
            node.value = self._ensure_node_in_anf(parent, field, node.value)
            return node
        if isinstance(node, (gast.Starred, gast.withitem, gast.slice)):
            return self._ensure_fields_in_anf(node, parent, field)
        if self._should_transform(parent, field, node):
            return self._do_transform_node(node)
        else:
            return node

    def _ensure_fields_in_anf(self, node, parent=None, super_field=None):
        for field in node._fields:
            if field.startswith('__'):
                continue
            parent_supplied = node if parent is None else parent
            field_supplied = field if super_field is None else super_field
            setattr(node, field, self._ensure_node_in_anf(parent_supplied, field_supplied, getattr(node, field)))
        return node

    def _visit_strict_statement(self, node, children_ok_to_transform=True):
        assert not self._pending_statements
        node = self.generic_visit(node)
        if children_ok_to_transform:
            self._ensure_fields_in_anf(node)
        results = self._consume_pending_statements()
        results.append(node)
        return results

    def _visit_trivial_only_statement(self, node, msg):
        assert not self._pending_statements
        node = self.generic_visit(node)
        self._ensure_fields_in_anf(node)
        if self._pending_statements:
            raise ValueError(msg)
        else:
            return node

    def _visit_strict_expression(self, node):
        node = self.generic_visit(node)
        self._ensure_fields_in_anf(node)
        return node

    def _visit_trivial_only_expression(self, node, msg):
        k = len(self._pending_statements)
        node = self.generic_visit(node)
        self._ensure_fields_in_anf(node)
        if len(self._pending_statements) != k:
            raise ValueError(msg)
        else:
            return node

    def visit_Return(self, node):
        return self._visit_strict_statement(node)

    def visit_Delete(self, node):
        return self._visit_strict_statement(node, children_ok_to_transform=False)

    def visit_Assign(self, node):
        return self._visit_strict_statement(node, children_ok_to_transform=False)

    def visit_AugAssign(self, node):
        return self._visit_strict_statement(node, children_ok_to_transform=False)

    def visit_Print(self, node):
        return self._visit_strict_statement(node)

    def visit_For(self, node):
        assert not self._pending_statements
        self.visit(node.iter)
        node.iter = self._ensure_node_in_anf(node, 'iter', node.iter)
        iter_stmts = self._consume_pending_statements()
        node = self.generic_visit(node)
        assert not self._pending_statements
        iter_stmts.append(node)
        return iter_stmts

    def visit_AsyncFor(self, node):
        msg = 'Nontrivial AsyncFor nodes not supported yet (need to think through the semantics).'
        return self._visit_trivial_only_statement(node, msg)

    def visit_While(self, node):
        assert not self._pending_statements
        self.visit(node.test)
        node.test = self._ensure_node_in_anf(node, 'test', node.test)
        if self._pending_statements:
            msg = 'While with nontrivial test not supported yet (need to avoid precomputing the test).'
            raise ValueError(msg)
        return self.generic_visit(node)

    def visit_If(self, node):
        assert not self._pending_statements
        self.visit(node.test)
        node.test = self._ensure_node_in_anf(node, 'test', node.test)
        condition_stmts = self._consume_pending_statements()
        node = self.generic_visit(node)
        assert not self._pending_statements
        condition_stmts.append(node)
        return condition_stmts

    def visit_With(self, node):
        assert not self._pending_statements
        for item in node.items:
            self.visit(item)
        node.items = [self._ensure_node_in_anf(node, 'items', n) for n in node.items]
        contexts_stmts = self._consume_pending_statements()
        node = self.generic_visit(node)
        assert not self._pending_statements
        contexts_stmts.append(node)
        return contexts_stmts

    def visit_AsyncWith(self, node):
        msg = 'Nontrivial AsyncWith nodes not supported yet (need to think through the semantics).'
        return self._visit_trivial_only_statement(node, msg)

    def visit_Raise(self, node):
        return self._visit_strict_statement(node)

    def visit_Assert(self, node):
        msg = 'Nontrivial Assert nodes not supported yet (need to avoid computing the test when assertions are off, and avoid computing the irritant when the assertion does not fire).'
        return self._visit_trivial_only_statement(node, msg)

    def visit_Exec(self, node):
        return self._visit_strict_statement(node)

    def visit_Expr(self, node):
        return self._visit_strict_statement(node, children_ok_to_transform=False)

    def visit_BoolOp(self, node):
        msg = 'Nontrivial BoolOp nodes not supported yet (need to preserve short-circuiting semantics).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_BinOp(self, node):
        return self._visit_strict_expression(node)

    def visit_UnaryOp(self, node):
        return self._visit_strict_expression(node)

    def visit_Lambda(self, node):
        msg = 'Nontrivial Lambda nodes not supported (cannot insert statements into lambda bodies).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_IfExp(self, node):
        msg = 'Nontrivial IfExp nodes not supported yet (need to convert to If statement, to evaluate branches lazily and insert statements into them).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_Dict(self, node):
        return self._visit_strict_expression(node)

    def visit_Set(self, node):
        return self._visit_strict_expression(node)

    def visit_ListComp(self, node):
        msg = 'ListComp nodes not supported (need to convert to a form that tolerates assignment statements in clause bodies).'
        raise ValueError(msg)

    def visit_SetComp(self, node):
        msg = 'SetComp nodes not supported (need to convert to a form that tolerates assignment statements in clause bodies).'
        raise ValueError(msg)

    def visit_DictComp(self, node):
        msg = 'DictComp nodes not supported (need to convert to a form that tolerates assignment statements in clause bodies).'
        raise ValueError(msg)

    def visit_GeneratorExp(self, node):
        msg = 'GeneratorExp nodes not supported (need to convert to a form that tolerates assignment statements in clause bodies).'
        raise ValueError(msg)

    def visit_Await(self, node):
        msg = 'Nontrivial Await nodes not supported yet (need to think through the semantics).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_Yield(self, node):
        return self._visit_strict_expression(node)

    def visit_YieldFrom(self, node):
        msg = 'Nontrivial YieldFrom nodes not supported yet (need to unit-test them in Python 2).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_Compare(self, node):
        if len(node.ops) > 1:
            msg = 'Multi-ary compare nodes not supported yet (need to preserve short-circuiting semantics).'
            raise ValueError(msg)
        return self._visit_strict_expression(node)

    def visit_Call(self, node):
        return self._visit_strict_expression(node)

    def visit_Repr(self, node):
        msg = 'Nontrivial Repr nodes not supported yet (need to research their syntax and semantics).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_FormattedValue(self, node):
        msg = 'Nontrivial FormattedValue nodes not supported yet (need to unit-test them in Python 2).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_JoinedStr(self, node):
        msg = 'Nontrivial JoinedStr nodes not supported yet (need to unit-test them in Python 2).'
        return self._visit_trivial_only_expression(node, msg)

    def visit_Attribute(self, node):
        return self._visit_strict_expression(node)

    def visit_Subscript(self, node):
        return self._visit_strict_expression(node)

    def visit_List(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.ctx, gast.Store):
            self._ensure_fields_in_anf(node)
        return node

    def visit_Tuple(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.ctx, gast.Store):
            self._ensure_fields_in_anf(node)
        return node