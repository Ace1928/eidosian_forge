from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
class MarkParallelAssignments(EnvTransform):
    in_loop = False
    parallel_errors = False

    def __init__(self, context):
        self.parallel_block_stack = []
        super(MarkParallelAssignments, self).__init__(context)

    def mark_assignment(self, lhs, rhs, inplace_op=None):
        if isinstance(lhs, (ExprNodes.NameNode, Nodes.PyArgDeclNode)):
            if lhs.entry is None:
                return
            if self.parallel_block_stack:
                parallel_node = self.parallel_block_stack[-1]
                previous_assignment = parallel_node.assignments.get(lhs.entry)
                if previous_assignment:
                    pos, previous_inplace_op = previous_assignment
                    if inplace_op and previous_inplace_op and (inplace_op != previous_inplace_op):
                        t = (inplace_op, previous_inplace_op)
                        error(lhs.pos, "Reduction operator '%s' is inconsistent with previous reduction operator '%s'" % t)
                else:
                    pos = lhs.pos
                parallel_node.assignments[lhs.entry] = (pos, inplace_op)
                parallel_node.assigned_nodes.append(lhs)
        elif isinstance(lhs, ExprNodes.SequenceNode):
            for i, arg in enumerate(lhs.args):
                if not rhs or arg.is_starred:
                    item_node = None
                else:
                    item_node = rhs.inferable_item_node(i)
                self.mark_assignment(arg, item_node)
        else:
            pass

    def visit_WithTargetAssignmentStatNode(self, node):
        self.mark_assignment(node.lhs, node.with_node.enter_call)
        self.visitchildren(node)
        return node

    def visit_SingleAssignmentNode(self, node):
        self.mark_assignment(node.lhs, node.rhs)
        self.visitchildren(node)
        return node

    def visit_CascadedAssignmentNode(self, node):
        for lhs in node.lhs_list:
            self.mark_assignment(lhs, node.rhs)
        self.visitchildren(node)
        return node

    def visit_InPlaceAssignmentNode(self, node):
        self.mark_assignment(node.lhs, node.create_binop_node(), node.operator)
        self.visitchildren(node)
        return node

    def visit_ForInStatNode(self, node):
        is_special = False
        sequence = node.iterator.sequence
        target = node.target
        iterator_scope = node.iterator.expr_scope or self.current_env()
        if isinstance(sequence, ExprNodes.SimpleCallNode):
            function = sequence.function
            if sequence.self is None and function.is_name:
                entry = iterator_scope.lookup(function.name)
                if not entry or entry.is_builtin:
                    if function.name == 'reversed' and len(sequence.args) == 1:
                        sequence = sequence.args[0]
                    elif function.name == 'enumerate' and len(sequence.args) == 1:
                        if target.is_sequence_constructor and len(target.args) == 2:
                            iterator = sequence.args[0]
                            if iterator.is_name:
                                iterator_type = iterator.infer_type(iterator_scope)
                                if iterator_type.is_builtin_type:
                                    self.mark_assignment(target.args[0], ExprNodes.IntNode(target.pos, value='PY_SSIZE_T_MAX', type=PyrexTypes.c_py_ssize_t_type))
                                    target = target.args[1]
                                    sequence = sequence.args[0]
        if isinstance(sequence, ExprNodes.SimpleCallNode):
            function = sequence.function
            if sequence.self is None and function.is_name:
                entry = iterator_scope.lookup(function.name)
                if not entry or entry.is_builtin:
                    if function.name in ('range', 'xrange'):
                        is_special = True
                        for arg in sequence.args[:2]:
                            self.mark_assignment(target, arg)
                        if len(sequence.args) > 2:
                            self.mark_assignment(target, ExprNodes.binop_node(node.pos, '+', sequence.args[0], sequence.args[2]))
        if not is_special:
            self.mark_assignment(target, ExprNodes.IndexNode(node.pos, base=sequence, index=ExprNodes.IntNode(target.pos, value='PY_SSIZE_T_MAX', type=PyrexTypes.c_py_ssize_t_type)))
        self.visitchildren(node)
        return node

    def visit_ForFromStatNode(self, node):
        self.mark_assignment(node.target, node.bound1)
        if node.step is not None:
            self.mark_assignment(node.target, ExprNodes.binop_node(node.pos, '+', node.bound1, node.step))
        self.visitchildren(node)
        return node

    def visit_WhileStatNode(self, node):
        self.visitchildren(node)
        return node

    def visit_ExceptClauseNode(self, node):
        if node.target is not None:
            self.mark_assignment(node.target, object_expr)
        self.visitchildren(node)
        return node

    def visit_FromCImportStatNode(self, node):
        return node

    def visit_FromImportStatNode(self, node):
        for name, target in node.items:
            if name != '*':
                self.mark_assignment(target, object_expr)
        self.visitchildren(node)
        return node

    def visit_DefNode(self, node):
        if node.star_arg:
            self.mark_assignment(node.star_arg, TypedExprNode(Builtin.tuple_type, node.pos))
        if node.starstar_arg:
            self.mark_assignment(node.starstar_arg, TypedExprNode(Builtin.dict_type, node.pos))
        EnvTransform.visit_FuncDefNode(self, node)
        return node

    def visit_DelStatNode(self, node):
        for arg in node.args:
            self.mark_assignment(arg, arg)
        self.visitchildren(node)
        return node

    def visit_ParallelStatNode(self, node):
        if self.parallel_block_stack:
            node.parent = self.parallel_block_stack[-1]
        else:
            node.parent = None
        nested = False
        if node.is_prange:
            if not node.parent:
                node.is_parallel = True
            else:
                node.is_parallel = node.parent.is_prange or not node.parent.is_parallel
                nested = node.parent.is_prange
        else:
            node.is_parallel = True
            nested = node.parent and node.parent.is_prange
        self.parallel_block_stack.append(node)
        nested = nested or len(self.parallel_block_stack) > 2
        if not self.parallel_errors and nested and (not node.is_prange):
            error(node.pos, 'Only prange() may be nested')
            self.parallel_errors = True
        if node.is_prange:
            child_attrs = node.child_attrs
            node.child_attrs = ['body', 'target', 'args']
            self.visitchildren(node)
            node.child_attrs = child_attrs
            self.parallel_block_stack.pop()
            if node.else_clause:
                node.else_clause = self.visit(node.else_clause)
        else:
            self.visitchildren(node)
            self.parallel_block_stack.pop()
        self.parallel_errors = False
        return node

    def visit_YieldExprNode(self, node):
        if self.parallel_block_stack:
            error(node.pos, "'%s' not allowed in parallel sections" % node.expr_keyword)
        return node

    def visit_ReturnStatNode(self, node):
        node.in_parallel = bool(self.parallel_block_stack)
        return node