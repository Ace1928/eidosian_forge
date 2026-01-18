from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class SingleAssignmentNode(AssignmentNode):
    child_attrs = ['lhs', 'rhs']
    first = False
    is_overloaded_assignment = False
    is_assignment_expression = False
    declaration_only = False

    def analyse_declarations(self, env):
        from . import ExprNodes
        if isinstance(self.rhs, ExprNodes.CallNode):
            func_name = self.rhs.function.as_cython_attribute()
            if func_name:
                args, kwds = self.rhs.explicit_args_kwds()
                if func_name in ['declare', 'typedef']:
                    if len(args) > 2:
                        error(args[2].pos, 'Invalid positional argument.')
                        return
                    if kwds is not None:
                        kwdict = kwds.compile_time_value(None)
                        if func_name == 'typedef' or 'visibility' not in kwdict:
                            error(kwds.pos, 'Invalid keyword argument.')
                            return
                        visibility = kwdict['visibility']
                    else:
                        visibility = 'private'
                    type = args[0].analyse_as_type(env)
                    if type is None:
                        error(args[0].pos, 'Unknown type')
                        return
                    lhs = self.lhs
                    if func_name == 'declare':
                        if isinstance(lhs, ExprNodes.NameNode):
                            vars = [(lhs.name, lhs.pos)]
                        elif isinstance(lhs, ExprNodes.TupleNode):
                            vars = [(var.name, var.pos) for var in lhs.args]
                        else:
                            error(lhs.pos, 'Invalid declaration')
                            return
                        for var, pos in vars:
                            env.declare_var(var, type, pos, is_cdef=True, visibility=visibility)
                        if len(args) == 2:
                            self.rhs = args[1]
                        else:
                            self.declaration_only = True
                    else:
                        self.declaration_only = True
                        if not isinstance(lhs, ExprNodes.NameNode):
                            error(lhs.pos, 'Invalid declaration.')
                        env.declare_typedef(lhs.name, type, self.pos, visibility='private')
                elif func_name in ['struct', 'union']:
                    self.declaration_only = True
                    if len(args) > 0 or kwds is None:
                        error(self.rhs.pos, 'Struct or union members must be given by name.')
                        return
                    members = []
                    for member, type_node in kwds.key_value_pairs:
                        type = type_node.analyse_as_type(env)
                        if type is None:
                            error(type_node.pos, 'Unknown type')
                        else:
                            members.append((member.value, type, member.pos))
                    if len(members) < len(kwds.key_value_pairs):
                        return
                    if not isinstance(self.lhs, ExprNodes.NameNode):
                        error(self.lhs.pos, 'Invalid declaration.')
                    name = self.lhs.name
                    scope = StructOrUnionScope(name)
                    env.declare_struct_or_union(name, func_name, scope, False, self.rhs.pos)
                    for member, type, pos in members:
                        scope.declare_var(member, type, pos)
                elif func_name == 'fused_type':
                    self.declaration_only = True
                    if kwds:
                        error(self.rhs.function.pos, 'fused_type does not take keyword arguments')
                    fusednode = FusedTypeNode(self.rhs.pos, name=self.lhs.name, types=args)
                    fusednode.analyse_declarations(env)
        if self.declaration_only:
            return
        elif self.is_assignment_expression:
            self.lhs.analyse_assignment_expression_target_declaration(env)
        else:
            self.lhs.analyse_target_declaration(env)
            if (self.lhs.is_attribute or self.lhs.is_name) and self.lhs.entry and (not self.lhs.entry.known_standard_library_import):
                stdlib_import_name = self.rhs.get_known_standard_library_import()
                if stdlib_import_name:
                    self.lhs.entry.known_standard_library_import = stdlib_import_name

    def analyse_types(self, env, use_temp=0):
        from . import ExprNodes
        self.rhs = self.rhs.analyse_types(env)
        unrolled_assignment = self.unroll_rhs(env)
        if unrolled_assignment:
            return unrolled_assignment
        self.lhs = self.lhs.analyse_target_types(env)
        self.lhs.gil_assignment_check(env)
        unrolled_assignment = self.unroll_lhs(env)
        if unrolled_assignment:
            return unrolled_assignment
        if isinstance(self.lhs, ExprNodes.MemoryViewIndexNode):
            self.lhs.analyse_broadcast_operation(self.rhs)
            self.lhs = self.lhs.analyse_as_memview_scalar_assignment(self.rhs)
        elif self.lhs.type.is_array:
            if not isinstance(self.lhs, ExprNodes.SliceIndexNode):
                lhs = ExprNodes.SliceIndexNode(self.lhs.pos, base=self.lhs, start=None, stop=None)
                self.lhs = lhs.analyse_target_types(env)
        if self.lhs.type.is_cpp_class:
            op = env.lookup_operator_for_types(self.pos, '=', [self.lhs.type, self.rhs.type])
            if op:
                rhs = self.rhs
                self.is_overloaded_assignment = True
                self.exception_check = op.type.exception_check
                self.exception_value = op.type.exception_value
                if self.exception_check == '+' and self.exception_value is None:
                    env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
            else:
                rhs = self.rhs.coerce_to(self.lhs.type, env)
        else:
            rhs = self.rhs.coerce_to(self.lhs.type, env)
        if use_temp or rhs.is_attribute or (not rhs.is_name and (not rhs.is_literal) and rhs.type.is_pyobject):
            rhs = rhs.coerce_to_temp(env)
        elif rhs.type.is_pyobject:
            rhs = rhs.coerce_to_simple(env)
        self.rhs = rhs
        return self

    def unroll(self, node, target_size, env):
        from . import ExprNodes, UtilNodes
        base = node
        start_node = stop_node = step_node = check_node = None
        if node.type.is_ctuple:
            slice_size = node.type.size
        elif node.type.is_ptr or node.type.is_array:
            while isinstance(node, ExprNodes.SliceIndexNode) and (not (node.start or node.stop)):
                base = node = node.base
            if isinstance(node, ExprNodes.SliceIndexNode):
                base = node.base
                start_node = node.start
                if start_node:
                    start_node = start_node.coerce_to(PyrexTypes.c_py_ssize_t_type, env)
                stop_node = node.stop
                if stop_node:
                    stop_node = stop_node.coerce_to(PyrexTypes.c_py_ssize_t_type, env)
                elif node.type.is_array and node.type.size:
                    stop_node = ExprNodes.IntNode(self.pos, value=str(node.type.size), constant_result=node.type.size if isinstance(node.type.size, _py_int_types) else ExprNodes.constant_value_not_set)
                else:
                    error(self.pos, 'C array iteration requires known end index')
                    return
                step_node = None
                if step_node:
                    step_node = step_node.coerce_to(PyrexTypes.c_py_ssize_t_type, env)

                def get_const(node, none_value):
                    if node is None:
                        return none_value
                    elif node.has_constant_result():
                        return node.constant_result
                    else:
                        raise ValueError('Not a constant.')
                try:
                    slice_size = (get_const(stop_node, None) - get_const(start_node, 0)) / get_const(step_node, 1)
                except ValueError:
                    error(self.pos, 'C array assignment currently requires known endpoints')
                    return
            elif node.type.is_array:
                slice_size = node.type.size
                if not isinstance(slice_size, _py_int_types):
                    return
            else:
                return
        else:
            return
        if slice_size != target_size:
            error(self.pos, 'Assignment to/from slice of wrong length, expected %s, got %s' % (slice_size, target_size))
            return
        items = []
        base = UtilNodes.LetRefNode(base)
        refs = [base]
        if start_node and (not start_node.is_literal):
            start_node = UtilNodes.LetRefNode(start_node)
            refs.append(start_node)
        if stop_node and (not stop_node.is_literal):
            stop_node = UtilNodes.LetRefNode(stop_node)
            refs.append(stop_node)
        if step_node and (not step_node.is_literal):
            step_node = UtilNodes.LetRefNode(step_node)
            refs.append(step_node)
        for ix in range(target_size):
            ix_node = ExprNodes.IntNode(self.pos, value=str(ix), constant_result=ix, type=PyrexTypes.c_py_ssize_t_type)
            if step_node is not None:
                if step_node.has_constant_result():
                    step_value = ix_node.constant_result * step_node.constant_result
                    ix_node = ExprNodes.IntNode(self.pos, value=str(step_value), constant_result=step_value)
                else:
                    ix_node = ExprNodes.MulNode(self.pos, operator='*', operand1=step_node, operand2=ix_node)
            if start_node is not None:
                if start_node.has_constant_result() and ix_node.has_constant_result():
                    index_value = ix_node.constant_result + start_node.constant_result
                    ix_node = ExprNodes.IntNode(self.pos, value=str(index_value), constant_result=index_value)
                else:
                    ix_node = ExprNodes.AddNode(self.pos, operator='+', operand1=start_node, operand2=ix_node)
            items.append(ExprNodes.IndexNode(self.pos, base=base, index=ix_node.analyse_types(env)))
        return (check_node, refs, items)

    def unroll_assignments(self, refs, check_node, lhs_list, rhs_list, env):
        from . import UtilNodes
        assignments = []
        for lhs, rhs in zip(lhs_list, rhs_list):
            assignments.append(SingleAssignmentNode(self.pos, lhs=lhs, rhs=rhs, first=self.first))
        node = ParallelAssignmentNode(pos=self.pos, stats=assignments).analyse_expressions(env)
        if check_node:
            node = StatListNode(pos=self.pos, stats=[check_node, node])
        for ref in refs[::-1]:
            node = UtilNodes.LetNode(ref, node)
        return node

    def unroll_rhs(self, env):
        from . import ExprNodes
        if not isinstance(self.lhs, ExprNodes.TupleNode):
            return
        if any((arg.is_starred for arg in self.lhs.args)):
            return
        unrolled = self.unroll(self.rhs, len(self.lhs.args), env)
        if not unrolled:
            return
        check_node, refs, rhs = unrolled
        return self.unroll_assignments(refs, check_node, self.lhs.args, rhs, env)

    def unroll_lhs(self, env):
        if self.lhs.type.is_ctuple:
            return
        from . import ExprNodes
        if not isinstance(self.rhs, ExprNodes.TupleNode):
            return
        unrolled = self.unroll(self.lhs, len(self.rhs.args), env)
        if not unrolled:
            return
        check_node, refs, lhs = unrolled
        return self.unroll_assignments(refs, check_node, lhs, self.rhs.args, env)

    def generate_rhs_evaluation_code(self, code):
        self.rhs.generate_evaluation_code(code)

    def generate_assignment_code(self, code, overloaded_assignment=False):
        if self.is_overloaded_assignment:
            self.lhs.generate_assignment_code(self.rhs, code, overloaded_assignment=self.is_overloaded_assignment, exception_check=self.exception_check, exception_value=self.exception_value)
        else:
            self.lhs.generate_assignment_code(self.rhs, code)

    def generate_function_definitions(self, env, code):
        self.rhs.generate_function_definitions(env, code)

    def annotate(self, code):
        self.lhs.annotate(code)
        self.rhs.annotate(code)