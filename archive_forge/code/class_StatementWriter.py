from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
class StatementWriter(DeclarationWriter):
    """
    A Cython code writer for most language statement features.
    """

    def visit_SingleAssignmentNode(self, node):
        self.startline()
        self.visit(node.lhs)
        self.put(u' = ')
        self.visit(node.rhs)
        self.endline()

    def visit_CascadedAssignmentNode(self, node):
        self.startline()
        for lhs in node.lhs_list:
            self.visit(lhs)
            self.put(u' = ')
        self.visit(node.rhs)
        self.endline()

    def visit_PrintStatNode(self, node):
        self.startline(u'print ')
        self.comma_separated_list(node.arg_tuple.args)
        if not node.append_newline:
            self.put(u',')
        self.endline()

    def visit_ForInStatNode(self, node):
        self.startline(u'for ')
        if node.target.is_sequence_constructor:
            self.comma_separated_list(node.target.args)
        else:
            self.visit(node.target)
        self.put(u' in ')
        self.visit(node.iterator.sequence)
        self.endline(u':')
        self._visit_indented(node.body)
        if node.else_clause is not None:
            self.line(u'else:')
            self._visit_indented(node.else_clause)

    def visit_IfStatNode(self, node):
        self.startline(u'if ')
        self.visit(node.if_clauses[0].condition)
        self.endline(':')
        self._visit_indented(node.if_clauses[0].body)
        for clause in node.if_clauses[1:]:
            self.startline('elif ')
            self.visit(clause.condition)
            self.endline(':')
            self._visit_indented(clause.body)
        if node.else_clause is not None:
            self.line('else:')
            self._visit_indented(node.else_clause)

    def visit_WhileStatNode(self, node):
        self.startline(u'while ')
        self.visit(node.condition)
        self.endline(u':')
        self._visit_indented(node.body)
        if node.else_clause is not None:
            self.line('else:')
            self._visit_indented(node.else_clause)

    def visit_ContinueStatNode(self, node):
        self.line(u'continue')

    def visit_BreakStatNode(self, node):
        self.line(u'break')

    def visit_SequenceNode(self, node):
        self.comma_separated_list(node.args)

    def visit_ExprStatNode(self, node):
        self.startline()
        self.visit(node.expr)
        self.endline()

    def visit_InPlaceAssignmentNode(self, node):
        self.startline()
        self.visit(node.lhs)
        self.put(u' %s= ' % node.operator)
        self.visit(node.rhs)
        self.endline()

    def visit_WithStatNode(self, node):
        self.startline()
        self.put(u'with ')
        self.visit(node.manager)
        if node.target is not None:
            self.put(u' as ')
            self.visit(node.target)
        self.endline(u':')
        self._visit_indented(node.body)

    def visit_TryFinallyStatNode(self, node):
        self.line(u'try:')
        self._visit_indented(node.body)
        self.line(u'finally:')
        self._visit_indented(node.finally_clause)

    def visit_TryExceptStatNode(self, node):
        self.line(u'try:')
        self._visit_indented(node.body)
        for x in node.except_clauses:
            self.visit(x)
        if node.else_clause is not None:
            self.visit(node.else_clause)

    def visit_ExceptClauseNode(self, node):
        self.startline(u'except')
        if node.pattern is not None:
            self.put(u' ')
            self.visit(node.pattern)
        if node.target is not None:
            self.put(u', ')
            self.visit(node.target)
        self.endline(':')
        self._visit_indented(node.body)

    def visit_ReturnStatNode(self, node):
        self.startline('return')
        if node.value is not None:
            self.put(u' ')
            self.visit(node.value)
        self.endline()

    def visit_ReraiseStatNode(self, node):
        self.line('raise')

    def visit_ImportNode(self, node):
        self.put(u'(import %s)' % node.module_name.value)

    def visit_TempsBlockNode(self, node):
        """
        Temporaries are output like $1_1', where the first number is
        an index of the TempsBlockNode and the second number is an index
        of the temporary which that block allocates.
        """
        idx = 0
        for handle in node.temps:
            self.tempnames[handle] = '$%d_%d' % (self.tempblockindex, idx)
            idx += 1
        self.tempblockindex += 1
        self.visit(node.body)

    def visit_TempRefNode(self, node):
        self.put(self.tempnames[node.handle])