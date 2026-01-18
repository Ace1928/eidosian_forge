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
class TransformBuiltinMethods(EnvTransform):
    """
    Replace Cython's own cython.* builtins by the corresponding tree nodes.
    """

    def visit_SingleAssignmentNode(self, node):
        if node.declaration_only:
            return None
        else:
            self.visitchildren(node)
            return node

    def visit_AttributeNode(self, node):
        self.visitchildren(node)
        return self.visit_cython_attribute(node)

    def visit_NameNode(self, node):
        return self.visit_cython_attribute(node)

    def visit_cython_attribute(self, node):
        attribute = node.as_cython_attribute()
        if attribute:
            if attribute == u'__version__':
                from .. import __version__ as version
                node = ExprNodes.StringNode(node.pos, value=EncodedString(version))
            elif attribute == u'NULL':
                node = ExprNodes.NullNode(node.pos)
            elif attribute in (u'set', u'frozenset', u'staticmethod'):
                node = ExprNodes.NameNode(node.pos, name=EncodedString(attribute), entry=self.current_env().builtin_scope().lookup_here(attribute))
            elif PyrexTypes.parse_basic_type(attribute):
                pass
            elif self.context.cython_scope.lookup_qualified_name(attribute):
                pass
            else:
                error(node.pos, u"'%s' not a valid cython attribute or is being used incorrectly" % attribute)
        return node

    def visit_ExecStatNode(self, node):
        lenv = self.current_env()
        self.visitchildren(node)
        if len(node.args) == 1:
            node.args.append(ExprNodes.GlobalsExprNode(node.pos))
            if not lenv.is_module_scope:
                node.args.append(ExprNodes.LocalsExprNode(node.pos, self.current_scope_node(), lenv))
        return node

    def _inject_locals(self, node, func_name):
        lenv = self.current_env()
        entry = lenv.lookup_here(func_name)
        if entry:
            return node
        pos = node.pos
        if func_name in ('locals', 'vars'):
            if func_name == 'locals' and len(node.args) > 0:
                error(self.pos, "Builtin 'locals()' called with wrong number of args, expected 0, got %d" % len(node.args))
                return node
            elif func_name == 'vars':
                if len(node.args) > 1:
                    error(self.pos, "Builtin 'vars()' called with wrong number of args, expected 0-1, got %d" % len(node.args))
                if len(node.args) > 0:
                    return node
            return ExprNodes.LocalsExprNode(pos, self.current_scope_node(), lenv)
        else:
            if len(node.args) > 1:
                error(self.pos, "Builtin 'dir()' called with wrong number of args, expected 0-1, got %d" % len(node.args))
            if len(node.args) > 0:
                return node
            if lenv.is_py_class_scope or lenv.is_module_scope:
                if lenv.is_py_class_scope:
                    pyclass = self.current_scope_node()
                    locals_dict = ExprNodes.CloneNode(pyclass.dict)
                else:
                    locals_dict = ExprNodes.GlobalsExprNode(pos)
                return ExprNodes.SortedDictKeysNode(locals_dict)
            local_names = sorted((var.name for var in lenv.entries.values() if var.name))
            items = [ExprNodes.IdentifierStringNode(pos, value=var) for var in local_names]
            return ExprNodes.ListNode(pos, args=items)

    def visit_PrimaryCmpNode(self, node):
        self.visitchildren(node)
        if node.operator in 'not_in':
            if isinstance(node.operand2, ExprNodes.SortedDictKeysNode):
                arg = node.operand2.arg
                if isinstance(arg, ExprNodes.NoneCheckNode):
                    arg = arg.arg
                node.operand2 = arg
        return node

    def visit_CascadedCmpNode(self, node):
        return self.visit_PrimaryCmpNode(node)

    def _inject_eval(self, node, func_name):
        lenv = self.current_env()
        entry = lenv.lookup(func_name)
        if len(node.args) != 1 or (entry and (not entry.is_builtin)):
            return node
        node.args.append(ExprNodes.GlobalsExprNode(node.pos))
        if not lenv.is_module_scope:
            node.args.append(ExprNodes.LocalsExprNode(node.pos, self.current_scope_node(), lenv))
        return node

    def _inject_super(self, node, func_name):
        lenv = self.current_env()
        entry = lenv.lookup_here(func_name)
        if entry or node.args:
            return node
        def_node = self.current_scope_node()
        if not isinstance(def_node, Nodes.DefNode) or not def_node.args or len(self.env_stack) < 2:
            return node
        class_node, class_scope = self.env_stack[-2]
        if class_scope.is_py_class_scope:
            def_node.requires_classobj = True
            class_node.class_cell.is_active = True
            node.args = [ExprNodes.ClassCellNode(node.pos, is_generator=def_node.is_generator), ExprNodes.NameNode(node.pos, name=def_node.args[0].name)]
        elif class_scope.is_c_class_scope:
            node.args = [ExprNodes.NameNode(node.pos, name=class_node.scope.name, entry=class_node.entry), ExprNodes.NameNode(node.pos, name=def_node.args[0].name)]
        return node

    def visit_SimpleCallNode(self, node):
        function = node.function.as_cython_attribute()
        if function:
            if function in InterpretCompilerDirectives.unop_method_nodes:
                if len(node.args) != 1:
                    error(node.function.pos, u'%s() takes exactly one argument' % function)
                else:
                    node = InterpretCompilerDirectives.unop_method_nodes[function](node.function.pos, operand=node.args[0])
            elif function in InterpretCompilerDirectives.binop_method_nodes:
                if len(node.args) != 2:
                    error(node.function.pos, u'%s() takes exactly two arguments' % function)
                else:
                    node = InterpretCompilerDirectives.binop_method_nodes[function](node.function.pos, operand1=node.args[0], operand2=node.args[1])
            elif function == u'cast':
                if len(node.args) != 2:
                    error(node.function.pos, u'cast() takes exactly two arguments and an optional typecheck keyword')
                else:
                    type = node.args[0].analyse_as_type(self.current_env())
                    if type:
                        node = ExprNodes.TypecastNode(node.function.pos, type=type, operand=node.args[1], typecheck=False)
                    else:
                        error(node.args[0].pos, 'Not a type')
            elif function == u'sizeof':
                if len(node.args) != 1:
                    error(node.function.pos, u'sizeof() takes exactly one argument')
                else:
                    type = node.args[0].analyse_as_type(self.current_env())
                    if type:
                        node = ExprNodes.SizeofTypeNode(node.function.pos, arg_type=type)
                    else:
                        node = ExprNodes.SizeofVarNode(node.function.pos, operand=node.args[0])
            elif function == 'cmod':
                if len(node.args) != 2:
                    error(node.function.pos, u'cmod() takes exactly two arguments')
                else:
                    node = ExprNodes.binop_node(node.function.pos, '%', node.args[0], node.args[1])
                    node.cdivision = True
            elif function == 'cdiv':
                if len(node.args) != 2:
                    error(node.function.pos, u'cdiv() takes exactly two arguments')
                else:
                    node = ExprNodes.binop_node(node.function.pos, '/', node.args[0], node.args[1])
                    node.cdivision = True
            elif function == u'set':
                node.function = ExprNodes.NameNode(node.pos, name=EncodedString('set'))
            elif function == u'staticmethod':
                node.function = ExprNodes.NameNode(node.pos, name=EncodedString('staticmethod'))
            elif self.context.cython_scope.lookup_qualified_name(function):
                pass
            else:
                error(node.function.pos, u"'%s' not a valid cython language construct" % function)
        self.visitchildren(node)
        if isinstance(node, ExprNodes.SimpleCallNode) and node.function.is_name:
            func_name = node.function.name
            if func_name in ('dir', 'locals', 'vars'):
                return self._inject_locals(node, func_name)
            if func_name == 'eval':
                return self._inject_eval(node, func_name)
            if func_name == 'super':
                return self._inject_super(node, func_name)
        return node

    def visit_GeneralCallNode(self, node):
        function = node.function.as_cython_attribute()
        if function == u'cast':
            args = node.positional_args.args
            kwargs = node.keyword_args.compile_time_value(None)
            if len(args) != 2 or len(kwargs) > 1 or (len(kwargs) == 1 and 'typecheck' not in kwargs):
                error(node.function.pos, u'cast() takes exactly two arguments and an optional typecheck keyword')
            else:
                type = args[0].analyse_as_type(self.current_env())
                if type:
                    typecheck = kwargs.get('typecheck', False)
                    node = ExprNodes.TypecastNode(node.function.pos, type=type, operand=args[1], typecheck=typecheck)
                else:
                    error(args[0].pos, 'Not a type')
        self.visitchildren(node)
        return node