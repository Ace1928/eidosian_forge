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
class AnalyseDeclarationsTransform(EnvTransform):
    basic_property = TreeFragment(u'\nproperty NAME:\n    def __get__(self):\n        return ATTR\n    def __set__(self, value):\n        ATTR = value\n    ', level='c_class', pipeline=[NormalizeTree(None)])
    basic_pyobject_property = TreeFragment(u'\nproperty NAME:\n    def __get__(self):\n        return ATTR\n    def __set__(self, value):\n        ATTR = value\n    def __del__(self):\n        ATTR = None\n    ', level='c_class', pipeline=[NormalizeTree(None)])
    basic_property_ro = TreeFragment(u'\nproperty NAME:\n    def __get__(self):\n        return ATTR\n    ', level='c_class', pipeline=[NormalizeTree(None)])
    struct_or_union_wrapper = TreeFragment(u'\ncdef class NAME:\n    cdef TYPE value\n    def __init__(self, MEMBER=None):\n        cdef int count\n        count = 0\n        INIT_ASSIGNMENTS\n        if IS_UNION and count > 1:\n            raise ValueError, "At most one union member should be specified."\n    def __str__(self):\n        return STR_FORMAT % MEMBER_TUPLE\n    def __repr__(self):\n        return REPR_FORMAT % MEMBER_TUPLE\n    ', pipeline=[NormalizeTree(None)])
    init_assignment = TreeFragment(u'\nif VALUE is not None:\n    ATTR = VALUE\n    count += 1\n    ', pipeline=[NormalizeTree(None)])
    fused_function = None
    in_lambda = 0

    def __call__(self, root):
        self.seen_vars_stack = []
        self.fused_error_funcs = set()
        super_class = super(AnalyseDeclarationsTransform, self)
        self._super_visit_FuncDefNode = super_class.visit_FuncDefNode
        return super_class.__call__(root)

    def visit_NameNode(self, node):
        self.seen_vars_stack[-1].add(node.name)
        return node

    def visit_ModuleNode(self, node):
        self.extra_module_declarations = []
        self.seen_vars_stack.append(set())
        node.analyse_declarations(self.current_env())
        self.visitchildren(node)
        self.seen_vars_stack.pop()
        node.body.stats.extend(self.extra_module_declarations)
        return node

    def visit_LambdaNode(self, node):
        self.in_lambda += 1
        node.analyse_declarations(self.current_env())
        self.visitchildren(node)
        self.in_lambda -= 1
        return node

    def visit_CClassDefNode(self, node):
        node = self.visit_ClassDefNode(node)
        if node.scope and 'dataclasses.dataclass' in node.scope.directives:
            from .Dataclass import handle_cclass_dataclass
            handle_cclass_dataclass(node, node.scope.directives['dataclasses.dataclass'], self)
        if node.scope and node.scope.implemented and node.body:
            stats = []
            for entry in node.scope.var_entries:
                if entry.needs_property:
                    property = self.create_Property(entry)
                    property.analyse_declarations(node.scope)
                    self.visit(property)
                    stats.append(property)
            if stats:
                node.body.stats += stats
            if node.visibility != 'extern' and (not node.scope.lookup('__reduce__')) and (not node.scope.lookup('__reduce_ex__')):
                self._inject_pickle_methods(node)
        return node

    def _inject_pickle_methods(self, node):
        env = self.current_env()
        if node.scope.directives['auto_pickle'] is False:
            return
        auto_pickle_forced = node.scope.directives['auto_pickle'] is True
        all_members = []
        cls = node.entry.type
        cinit = None
        inherited_reduce = None
        while cls is not None:
            all_members.extend((e for e in cls.scope.var_entries if e.name not in ('__weakref__', '__dict__')))
            cinit = cinit or cls.scope.lookup('__cinit__')
            inherited_reduce = inherited_reduce or cls.scope.lookup('__reduce__') or cls.scope.lookup('__reduce_ex__')
            cls = cls.base_type
        all_members.sort(key=lambda e: e.name)
        if inherited_reduce:
            return
        non_py = [e for e in all_members if not e.type.is_pyobject and (not e.type.can_coerce_to_pyobject(env) or not e.type.can_coerce_from_pyobject(env))]
        structs = [e for e in all_members if e.type.is_struct_or_union]
        if cinit or non_py or (structs and (not auto_pickle_forced)):
            if cinit:
                msg = 'no default __reduce__ due to non-trivial __cinit__'
            elif non_py:
                msg = '%s cannot be converted to a Python object for pickling' % ','.join(('self.%s' % e.name for e in non_py))
            else:
                msg = 'Pickling of struct members such as %s must be explicitly requested with @auto_pickle(True)' % ','.join(('self.%s' % e.name for e in structs))
            if auto_pickle_forced:
                error(node.pos, msg)
            pickle_func = TreeFragment(u'\n                def __reduce_cython__(self):\n                    raise TypeError, "%(msg)s"\n                def __setstate_cython__(self, __pyx_state):\n                    raise TypeError, "%(msg)s"\n                ' % {'msg': msg}, level='c_class', pipeline=[NormalizeTree(None)]).substitute({})
            pickle_func.analyse_declarations(node.scope)
            self.visit(pickle_func)
            node.body.stats.append(pickle_func)
        else:
            for e in all_members:
                if not e.type.is_pyobject:
                    e.type.create_to_py_utility_code(env)
                    e.type.create_from_py_utility_code(env)
            all_members_names = [e.name for e in all_members]
            checksums = _calculate_pickle_checksums(all_members_names)
            unpickle_func_name = '__pyx_unpickle_%s' % node.punycode_class_name
            unpickle_func = TreeFragment(u'\n                def %(unpickle_func_name)s(__pyx_type, long __pyx_checksum, __pyx_state):\n                    cdef object __pyx_PickleError\n                    cdef object __pyx_result\n                    if __pyx_checksum not in %(checksums)s:\n                        from pickle import PickleError as __pyx_PickleError\n                        raise __pyx_PickleError, "Incompatible checksums (0x%%x vs %(checksums)s = (%(members)s))" %% __pyx_checksum\n                    __pyx_result = %(class_name)s.__new__(__pyx_type)\n                    if __pyx_state is not None:\n                        %(unpickle_func_name)s__set_state(<%(class_name)s> __pyx_result, __pyx_state)\n                    return __pyx_result\n\n                cdef %(unpickle_func_name)s__set_state(%(class_name)s __pyx_result, tuple __pyx_state):\n                    %(assignments)s\n                    if len(__pyx_state) > %(num_members)d and hasattr(__pyx_result, \'__dict__\'):\n                        __pyx_result.__dict__.update(__pyx_state[%(num_members)d])\n                ' % {'unpickle_func_name': unpickle_func_name, 'checksums': '(%s)' % ', '.join(checksums), 'members': ', '.join(all_members_names), 'class_name': node.class_name, 'assignments': '; '.join(('__pyx_result.%s = __pyx_state[%s]' % (v, ix) for ix, v in enumerate(all_members_names))), 'num_members': len(all_members_names)}, level='module', pipeline=[NormalizeTree(None)]).substitute({})
            unpickle_func.analyse_declarations(node.entry.scope)
            self.visit(unpickle_func)
            self.extra_module_declarations.append(unpickle_func)
            pickle_func = TreeFragment(u"\n                def __reduce_cython__(self):\n                    cdef tuple state\n                    cdef object _dict\n                    cdef bint use_setstate\n                    state = (%(members)s)\n                    _dict = getattr(self, '__dict__', None)\n                    if _dict is not None:\n                        state += (_dict,)\n                        use_setstate = True\n                    else:\n                        use_setstate = %(any_notnone_members)s\n                    if use_setstate:\n                        return %(unpickle_func_name)s, (type(self), %(checksum)s, None), state\n                    else:\n                        return %(unpickle_func_name)s, (type(self), %(checksum)s, state)\n\n                def __setstate_cython__(self, __pyx_state):\n                    %(unpickle_func_name)s__set_state(self, __pyx_state)\n                " % {'unpickle_func_name': unpickle_func_name, 'checksum': checksums[0], 'members': ', '.join(('self.%s' % v for v in all_members_names)) + (',' if len(all_members_names) == 1 else ''), 'any_notnone_members': ' or '.join(['self.%s is not None' % e.name for e in all_members if e.type.is_pyobject] or ['False'])}, level='c_class', pipeline=[NormalizeTree(None)]).substitute({})
            pickle_func.analyse_declarations(node.scope)
            self.enter_scope(node, node.scope)
            self.visit(pickle_func)
            self.exit_scope()
            node.body.stats.append(pickle_func)

    def _handle_fused_def_decorators(self, old_decorators, env, node):
        """
        Create function calls to the decorators and reassignments to
        the function.
        """
        decorators = []
        for decorator in old_decorators:
            func = decorator.decorator
            if not func.is_name or func.name not in ('staticmethod', 'classmethod') or env.lookup_here(func.name):
                decorators.append(decorator)
        if decorators:
            transform = DecoratorTransform(self.context)
            def_node = node.node
            _, reassignments = transform.chain_decorators(def_node, decorators, def_node.name)
            reassignments.analyse_declarations(env)
            node = [node, reassignments]
        return node

    def _handle_def(self, decorators, env, node):
        """Handle def or cpdef fused functions"""
        node.stats.insert(0, node.py_func)
        self.visitchild(node, 'py_func')
        node.update_fused_defnode_entry(env)
        node.py_func.entry.signature.use_fastcall = False
        pycfunc = ExprNodes.PyCFunctionNode.from_defnode(node.py_func, binding=True)
        pycfunc = ExprNodes.ProxyNode(pycfunc.coerce_to_temp(env))
        node.resulting_fused_function = pycfunc
        node.fused_func_assignment = self._create_assignment(node.py_func, ExprNodes.CloneNode(pycfunc), env)
        if decorators:
            node = self._handle_fused_def_decorators(decorators, env, node)
        return node

    def _create_fused_function(self, env, node):
        """Create a fused function for a DefNode with fused arguments"""
        from . import FusedNode
        if self.fused_function or self.in_lambda:
            if self.fused_function not in self.fused_error_funcs:
                if self.in_lambda:
                    error(node.pos, 'Fused lambdas not allowed')
                else:
                    error(node.pos, 'Cannot nest fused functions')
            self.fused_error_funcs.add(self.fused_function)
            node.body = Nodes.PassStatNode(node.pos)
            for arg in node.args:
                if arg.type.is_fused:
                    arg.type = arg.type.get_fused_types()[0]
            return node
        decorators = getattr(node, 'decorators', None)
        node = FusedNode.FusedCFuncDefNode(node, env)
        self.fused_function = node
        self.visitchildren(node)
        self.fused_function = None
        if node.py_func:
            node = self._handle_def(decorators, env, node)
        return node

    def _handle_fused(self, node):
        if node.is_generator and node.has_fused_arguments:
            node.has_fused_arguments = False
            error(node.pos, 'Fused generators not supported')
            node.gbody = Nodes.StatListNode(node.pos, stats=[], body=Nodes.PassStatNode(node.pos))
        return node.has_fused_arguments

    def visit_FuncDefNode(self, node):
        """
        Analyse a function and its body, as that hasn't happened yet.  Also
        analyse the directive_locals set by @cython.locals().

        Then, if we are a function with fused arguments, replace the function
        (after it has declared itself in the symbol table!) with a
        FusedCFuncDefNode, and analyse its children (which are in turn normal
        functions). If we're a normal function, just analyse the body of the
        function.
        """
        env = self.current_env()
        self.seen_vars_stack.append(set())
        lenv = node.local_scope
        node.declare_arguments(lenv)
        for var, type_node in node.directive_locals.items():
            if not lenv.lookup_here(var):
                type = type_node.analyse_as_type(lenv)
                if type and type.is_fused and lenv.fused_to_specific:
                    type = type.specialize(lenv.fused_to_specific)
                if type:
                    lenv.declare_var(var, type, type_node.pos)
                else:
                    error(type_node.pos, 'Not a type')
        if self._handle_fused(node):
            node = self._create_fused_function(env, node)
        else:
            node.body.analyse_declarations(lenv)
            self._super_visit_FuncDefNode(node)
        self.seen_vars_stack.pop()
        if 'ufunc' in lenv.directives:
            from . import UFuncs
            return UFuncs.convert_to_ufunc(node)
        return node

    def visit_DefNode(self, node):
        node = self.visit_FuncDefNode(node)
        env = self.current_env()
        if not isinstance(node, Nodes.DefNode) or node.fused_py_func or node.is_generator_body or (not node.needs_assignment_synthesis(env)):
            return node
        return [node, self._synthesize_assignment(node, env)]

    def visit_GeneratorBodyDefNode(self, node):
        return self.visit_FuncDefNode(node)

    def _synthesize_assignment(self, node, env):
        genv = env
        while genv.is_py_class_scope or genv.is_c_class_scope:
            genv = genv.outer_scope
        if genv.is_closure_scope:
            rhs = node.py_cfunc_node = ExprNodes.InnerFunctionNode(node.pos, def_node=node, pymethdef_cname=node.entry.pymethdef_cname, code_object=ExprNodes.CodeObjectNode(node))
        else:
            binding = self.current_directives.get('binding')
            rhs = ExprNodes.PyCFunctionNode.from_defnode(node, binding)
            node.code_object = rhs.code_object
            if node.is_generator:
                node.gbody.code_object = node.code_object
        if env.is_py_class_scope:
            rhs.binding = True
        node.is_cyfunction = rhs.binding
        return self._create_assignment(node, rhs, env)

    def _create_assignment(self, def_node, rhs, env):
        if def_node.decorators:
            for decorator in def_node.decorators[::-1]:
                rhs = ExprNodes.SimpleCallNode(decorator.pos, function=decorator.decorator, args=[rhs])
            def_node.decorators = None
        assmt = Nodes.SingleAssignmentNode(def_node.pos, lhs=ExprNodes.NameNode(def_node.pos, name=def_node.name), rhs=rhs)
        assmt.analyse_declarations(env)
        return assmt

    def visit_func_outer_attrs(self, node):
        stack = self.seen_vars_stack.pop()
        super(AnalyseDeclarationsTransform, self).visit_func_outer_attrs(node)
        self.seen_vars_stack.append(stack)

    def visit_ScopedExprNode(self, node):
        env = self.current_env()
        node.analyse_declarations(env)
        if node.expr_scope:
            self.seen_vars_stack.append(set(self.seen_vars_stack[-1]))
            self.enter_scope(node, node.expr_scope)
            node.analyse_scoped_declarations(node.expr_scope)
            self.visitchildren(node)
            self.exit_scope()
            self.seen_vars_stack.pop()
        else:
            node.analyse_scoped_declarations(env)
            self.visitchildren(node)
        return node

    def visit_TempResultFromStatNode(self, node):
        self.visitchildren(node)
        node.analyse_declarations(self.current_env())
        return node

    def visit_CppClassNode(self, node):
        if node.visibility == 'extern':
            return None
        else:
            return self.visit_ClassDefNode(node)

    def visit_CStructOrUnionDefNode(self, node):
        if True:
            return None
        self_value = ExprNodes.AttributeNode(pos=node.pos, obj=ExprNodes.NameNode(pos=node.pos, name=u'self'), attribute=EncodedString(u'value'))
        var_entries = node.entry.type.scope.var_entries
        attributes = []
        for entry in var_entries:
            attributes.append(ExprNodes.AttributeNode(pos=entry.pos, obj=self_value, attribute=entry.name))
        init_assignments = []
        for entry, attr in zip(var_entries, attributes):
            init_assignments.append(self.init_assignment.substitute({u'VALUE': ExprNodes.NameNode(entry.pos, name=entry.name), u'ATTR': attr}, pos=entry.pos))
        str_format = u'%s(%s)' % (node.entry.type.name, ('%s, ' * len(attributes))[:-2])
        wrapper_class = self.struct_or_union_wrapper.substitute({u'INIT_ASSIGNMENTS': Nodes.StatListNode(node.pos, stats=init_assignments), u'IS_UNION': ExprNodes.BoolNode(node.pos, value=not node.entry.type.is_struct), u'MEMBER_TUPLE': ExprNodes.TupleNode(node.pos, args=attributes), u'STR_FORMAT': ExprNodes.StringNode(node.pos, value=EncodedString(str_format)), u'REPR_FORMAT': ExprNodes.StringNode(node.pos, value=EncodedString(str_format.replace('%s', '%r')))}, pos=node.pos).stats[0]
        wrapper_class.class_name = node.name
        wrapper_class.shadow = True
        class_body = wrapper_class.body.stats
        assert isinstance(class_body[0].base_type, Nodes.CSimpleBaseTypeNode)
        class_body[0].base_type.name = node.name
        init_method = class_body[1]
        assert isinstance(init_method, Nodes.DefNode) and init_method.name == '__init__'
        arg_template = init_method.args[1]
        if not node.entry.type.is_struct:
            arg_template.kw_only = True
        del init_method.args[1]
        for entry, attr in zip(var_entries, attributes):
            arg = copy.deepcopy(arg_template)
            arg.declarator.name = entry.name
            init_method.args.append(arg)
        for entry, attr in zip(var_entries, attributes):
            if entry.type.is_pyobject:
                template = self.basic_pyobject_property
            else:
                template = self.basic_property
            property = template.substitute({u'ATTR': attr}, pos=entry.pos).stats[0]
            property.name = entry.name
            wrapper_class.body.stats.append(property)
        wrapper_class.analyse_declarations(self.current_env())
        return self.visit_CClassDefNode(wrapper_class)

    def visit_CDeclaratorNode(self, node):
        self.visitchildren(node)
        return node

    def visit_CTypeDefNode(self, node):
        return node

    def visit_CBaseTypeNode(self, node):
        return None

    def visit_CEnumDefNode(self, node):
        if node.visibility == 'public':
            return node
        else:
            return None

    def visit_CNameDeclaratorNode(self, node):
        if node.name in self.seen_vars_stack[-1]:
            entry = self.current_env().lookup(node.name)
            if entry is None or (entry.visibility != 'extern' and (not entry.scope.is_c_class_scope)):
                error(node.pos, "cdef variable '%s' declared after it is used" % node.name)
        self.visitchildren(node)
        return node

    def visit_CVarDefNode(self, node):
        self.visitchildren(node)
        return None

    def visit_CnameDecoratorNode(self, node):
        child_node = self.visitchild(node, 'node')
        if not child_node:
            return None
        if type(child_node) is list:
            node.node = child_node[0]
            return [node] + child_node[1:]
        return node

    def create_Property(self, entry):
        if entry.visibility == 'public':
            if entry.type.is_pyobject:
                template = self.basic_pyobject_property
            else:
                template = self.basic_property
        elif entry.visibility == 'readonly':
            template = self.basic_property_ro
        property = template.substitute({u'ATTR': ExprNodes.AttributeNode(pos=entry.pos, obj=ExprNodes.NameNode(pos=entry.pos, name='self'), attribute=entry.name)}, pos=entry.pos).stats[0]
        property.name = entry.name
        property.doc = entry.doc
        return property

    def visit_AssignmentExpressionNode(self, node):
        self.visitchildren(node)
        node.analyse_declarations(self.current_env())
        return node