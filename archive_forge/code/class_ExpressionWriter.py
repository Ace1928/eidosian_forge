from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
class ExpressionWriter(TreeVisitor):
    """
    A Cython code writer that is intentionally limited to expressions.
    """

    def __init__(self, result=None):
        super(ExpressionWriter, self).__init__()
        if result is None:
            result = u''
        self.result = result
        self.precedence = [0]

    def write(self, tree):
        self.visit(tree)
        return self.result

    def put(self, s):
        self.result += s

    def remove(self, s):
        if self.result.endswith(s):
            self.result = self.result[:-len(s)]

    def comma_separated_list(self, items):
        if len(items) > 0:
            for item in items[:-1]:
                self.visit(item)
                self.put(u', ')
            self.visit(items[-1])

    def visit_Node(self, node):
        raise AssertionError('Node not handled by serializer: %r' % node)

    def visit_IntNode(self, node):
        self.put(node.value)

    def visit_FloatNode(self, node):
        self.put(node.value)

    def visit_NoneNode(self, node):
        self.put(u'None')

    def visit_NameNode(self, node):
        self.put(node.name)

    def visit_EllipsisNode(self, node):
        self.put(u'...')

    def visit_BoolNode(self, node):
        self.put(str(node.value))

    def visit_ConstNode(self, node):
        self.put(str(node.value))

    def visit_ImagNode(self, node):
        self.put(node.value)
        self.put(u'j')

    def emit_string(self, node, prefix=u''):
        repr_val = repr(node.value)
        if repr_val[0] in 'ub':
            repr_val = repr_val[1:]
        self.put(u'%s%s' % (prefix, repr_val))

    def visit_BytesNode(self, node):
        self.emit_string(node, u'b')

    def visit_StringNode(self, node):
        self.emit_string(node)

    def visit_UnicodeNode(self, node):
        self.emit_string(node, u'u')

    def emit_sequence(self, node, parens=(u'', u'')):
        open_paren, close_paren = parens
        items = node.subexpr_nodes()
        self.put(open_paren)
        self.comma_separated_list(items)
        self.put(close_paren)

    def visit_ListNode(self, node):
        self.emit_sequence(node, u'[]')

    def visit_TupleNode(self, node):
        self.emit_sequence(node, u'()')

    def visit_SetNode(self, node):
        if len(node.subexpr_nodes()) > 0:
            self.emit_sequence(node, u'{}')
        else:
            self.put(u'set()')

    def visit_DictNode(self, node):
        self.emit_sequence(node, u'{}')

    def visit_DictItemNode(self, node):
        self.visit(node.key)
        self.put(u': ')
        self.visit(node.value)
    unop_precedence = {'not': 3, '!': 3, '+': 11, '-': 11, '~': 11}
    binop_precedence = {'or': 1, 'and': 2, 'in': 4, 'not_in': 4, 'is': 4, 'is_not': 4, '<': 4, '<=': 4, '>': 4, '>=': 4, '!=': 4, '==': 4, '|': 5, '^': 6, '&': 7, '<<': 8, '>>': 8, '+': 9, '-': 9, '*': 10, '@': 10, '/': 10, '//': 10, '%': 10, '**': 12}

    def operator_enter(self, new_prec):
        old_prec = self.precedence[-1]
        if old_prec > new_prec:
            self.put(u'(')
        self.precedence.append(new_prec)

    def operator_exit(self):
        old_prec, new_prec = self.precedence[-2:]
        if old_prec > new_prec:
            self.put(u')')
        self.precedence.pop()

    def visit_NotNode(self, node):
        op = 'not'
        prec = self.unop_precedence[op]
        self.operator_enter(prec)
        self.put(u'not ')
        self.visit(node.operand)
        self.operator_exit()

    def visit_UnopNode(self, node):
        op = node.operator
        prec = self.unop_precedence[op]
        self.operator_enter(prec)
        self.put(u'%s' % node.operator)
        self.visit(node.operand)
        self.operator_exit()

    def visit_BinopNode(self, node):
        op = node.operator
        prec = self.binop_precedence.get(op, 0)
        self.operator_enter(prec)
        self.visit(node.operand1)
        self.put(u' %s ' % op.replace('_', ' '))
        self.visit(node.operand2)
        self.operator_exit()

    def visit_BoolBinopNode(self, node):
        self.visit_BinopNode(node)

    def visit_PrimaryCmpNode(self, node):
        self.visit_BinopNode(node)

    def visit_IndexNode(self, node):
        self.visit(node.base)
        self.put(u'[')
        if isinstance(node.index, TupleNode):
            if node.index.subexpr_nodes():
                self.emit_sequence(node.index)
            else:
                self.put(u'()')
        else:
            self.visit(node.index)
        self.put(u']')

    def visit_SliceIndexNode(self, node):
        self.visit(node.base)
        self.put(u'[')
        if node.start:
            self.visit(node.start)
        self.put(u':')
        if node.stop:
            self.visit(node.stop)
        if node.slice:
            self.put(u':')
            self.visit(node.slice)
        self.put(u']')

    def visit_SliceNode(self, node):
        if not node.start.is_none:
            self.visit(node.start)
        self.put(u':')
        if not node.stop.is_none:
            self.visit(node.stop)
        if not node.step.is_none:
            self.put(u':')
            self.visit(node.step)

    def visit_CondExprNode(self, node):
        self.visit(node.true_val)
        self.put(u' if ')
        self.visit(node.test)
        self.put(u' else ')
        self.visit(node.false_val)

    def visit_AttributeNode(self, node):
        self.visit(node.obj)
        self.put(u'.%s' % node.attribute)

    def visit_SimpleCallNode(self, node):
        self.visit(node.function)
        self.put(u'(')
        self.comma_separated_list(node.args)
        self.put(')')

    def emit_pos_args(self, node):
        if node is None:
            return
        if isinstance(node, AddNode):
            self.emit_pos_args(node.operand1)
            self.emit_pos_args(node.operand2)
        elif isinstance(node, TupleNode):
            for expr in node.subexpr_nodes():
                self.visit(expr)
                self.put(u', ')
        elif isinstance(node, AsTupleNode):
            self.put('*')
            self.visit(node.arg)
            self.put(u', ')
        else:
            self.visit(node)
            self.put(u', ')

    def emit_kwd_args(self, node):
        if node is None:
            return
        if isinstance(node, MergedDictNode):
            for expr in node.subexpr_nodes():
                self.emit_kwd_args(expr)
        elif isinstance(node, DictNode):
            for expr in node.subexpr_nodes():
                self.put(u'%s=' % expr.key.value)
                self.visit(expr.value)
                self.put(u', ')
        else:
            self.put(u'**')
            self.visit(node)
            self.put(u', ')

    def visit_GeneralCallNode(self, node):
        self.visit(node.function)
        self.put(u'(')
        self.emit_pos_args(node.positional_args)
        self.emit_kwd_args(node.keyword_args)
        self.remove(u', ')
        self.put(')')

    def emit_comprehension(self, body, target, sequence, condition, parens=(u'', u'')):
        open_paren, close_paren = parens
        self.put(open_paren)
        self.visit(body)
        self.put(u' for ')
        self.visit(target)
        self.put(u' in ')
        self.visit(sequence)
        if condition:
            self.put(u' if ')
            self.visit(condition)
        self.put(close_paren)

    def visit_ComprehensionAppendNode(self, node):
        self.visit(node.expr)

    def visit_DictComprehensionAppendNode(self, node):
        self.visit(node.key_expr)
        self.put(u': ')
        self.visit(node.value_expr)

    def visit_ComprehensionNode(self, node):
        tpmap = {'list': u'[]', 'dict': u'{}', 'set': u'{}'}
        parens = tpmap[node.type.py_type_name()]
        body = node.loop.body
        target = node.loop.target
        sequence = node.loop.iterator.sequence
        condition = None
        if hasattr(body, 'if_clauses'):
            condition = body.if_clauses[0].condition
            body = body.if_clauses[0].body
        self.emit_comprehension(body, target, sequence, condition, parens)

    def visit_GeneratorExpressionNode(self, node):
        body = node.loop.body
        target = node.loop.target
        sequence = node.loop.iterator.sequence
        condition = None
        if hasattr(body, 'if_clauses'):
            condition = body.if_clauses[0].condition
            body = body.if_clauses[0].body.expr.arg
        elif hasattr(body, 'expr'):
            body = body.expr.arg
        self.emit_comprehension(body, target, sequence, condition, u'()')