from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
class WithTransform(VisitorTransform, SkipDeclarations):

    def visit_WithStatNode(self, node):
        self.visitchildren(node, 'body')
        pos = node.pos
        is_async = node.is_async
        body, target, manager = (node.body, node.target, node.manager)
        manager = node.manager = ExprNodes.ProxyNode(manager)
        node.enter_call = ExprNodes.SimpleCallNode(pos, function=ExprNodes.AttributeNode(pos, obj=ExprNodes.CloneNode(manager), attribute=EncodedString('__aenter__' if is_async else '__enter__'), is_special_lookup=True), args=[], is_temp=True)
        if is_async:
            node.enter_call = ExprNodes.AwaitExprNode(pos, arg=node.enter_call)
        if target is not None:
            body = Nodes.StatListNode(pos, stats=[Nodes.WithTargetAssignmentStatNode(pos, lhs=target, with_node=node), body])
        excinfo_target = ExprNodes.TupleNode(pos, slow=True, args=[ExprNodes.ExcValueNode(pos) for _ in range(3)])
        except_clause = Nodes.ExceptClauseNode(pos, body=Nodes.IfStatNode(pos, if_clauses=[Nodes.IfClauseNode(pos, condition=ExprNodes.NotNode(pos, operand=ExprNodes.WithExitCallNode(pos, with_stat=node, test_if_run=False, args=excinfo_target, await_expr=ExprNodes.AwaitExprNode(pos, arg=None) if is_async else None)), body=Nodes.ReraiseStatNode(pos))], else_clause=None), pattern=None, target=None, excinfo_target=excinfo_target)
        node.body = Nodes.TryFinallyStatNode(pos, body=Nodes.TryExceptStatNode(pos, body=body, except_clauses=[except_clause], else_clause=None), finally_clause=Nodes.ExprStatNode(pos, expr=ExprNodes.WithExitCallNode(pos, with_stat=node, test_if_run=True, args=ExprNodes.TupleNode(pos, args=[ExprNodes.NoneNode(pos) for _ in range(3)]), await_expr=ExprNodes.AwaitExprNode(pos, arg=None) if is_async else None)), handle_error_case=False)
        return node

    def visit_ExprNode(self, node):
        return node
    visit_Node = VisitorTransform.recurse_to_children