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
class CFuncDeclaratorNode(CDeclaratorNode):
    child_attrs = ['base', 'args', 'exception_value']
    overridable = 0
    optional_arg_count = 0
    is_const_method = 0
    templates = None

    def declared_name(self):
        return self.base.declared_name()

    def analyse_templates(self):
        if isinstance(self.base, CArrayDeclaratorNode):
            from .ExprNodes import TupleNode, NameNode
            template_node = self.base.dimension
            if isinstance(template_node, TupleNode):
                template_nodes = template_node.args
            elif isinstance(template_node, NameNode):
                template_nodes = [template_node]
            else:
                error(template_node.pos, 'Template arguments must be a list of names')
                return None
            self.templates = []
            for template in template_nodes:
                if isinstance(template, NameNode):
                    self.templates.append(PyrexTypes.TemplatePlaceholderType(template.name))
                else:
                    error(template.pos, 'Template arguments must be a list of names')
            self.base = self.base.base
            return self.templates
        else:
            return None

    def analyse(self, return_type, env, nonempty=0, directive_locals=None, visibility=None, in_pxd=False):
        if directive_locals is None:
            directive_locals = {}
        if nonempty:
            nonempty -= 1
        func_type_args = []
        for i, arg_node in enumerate(self.args):
            name_declarator, type = arg_node.analyse(env, nonempty=nonempty, is_self_arg=i == 0 and env.is_c_class_scope and ('staticmethod' not in env.directives))
            name = name_declarator.name
            if name in directive_locals:
                type_node = directive_locals[name]
                other_type = type_node.analyse_as_type(env)
                if other_type is None:
                    error(type_node.pos, 'Not a type')
                elif type is not PyrexTypes.py_object_type and (not type.same_as(other_type)):
                    error(self.base.pos, 'Signature does not agree with previous declaration')
                    error(type_node.pos, 'Previous declaration here')
                else:
                    type = other_type
            if name_declarator.cname:
                error(self.pos, 'Function argument cannot have C name specification')
            if i == 0 and env.is_c_class_scope and type.is_unspecified:
                type = env.parent_type
            if type.is_array:
                type = PyrexTypes.c_ptr_type(type.base_type)
            if type.is_void:
                error(arg_node.pos, 'Use spam() rather than spam(void) to declare a function with no arguments.')
            func_type_args.append(PyrexTypes.CFuncTypeArg(name, type, arg_node.pos))
            if arg_node.default:
                self.optional_arg_count += 1
            elif self.optional_arg_count:
                error(self.pos, 'Non-default argument follows default argument')
        exc_val = None
        exc_check = 0
        if env.directives['legacy_implicit_noexcept'] and (not return_type.is_pyobject) and (not self.has_explicit_exc_clause) and self.exception_check and (visibility != 'extern'):
            self.exception_check = False
            warning(self.pos, "Implicit noexcept declaration is deprecated. Function declaration should contain 'noexcept' keyword.", level=2)
        if self.exception_check == '+':
            env.add_include_file('ios')
            env.add_include_file('new')
            env.add_include_file('stdexcept')
            env.add_include_file('typeinfo')
        elif return_type.is_pyobject and self.exception_check:
            self.exception_check = False
        if return_type.is_pyobject and (self.exception_value or self.exception_check) and (self.exception_check != '+'):
            error(self.pos, 'Exception clause not allowed for function returning Python object')
        elif return_type.is_pyobject and (not self.exception_check) and (visibility != 'extern') and self.has_explicit_exc_clause:
            warning(self.pos, 'noexcept clause is ignored for function returning Python object', 1)
        else:
            if self.exception_value is None and self.exception_check and (self.exception_check != '+'):
                if return_type.exception_value is not None and (visibility != 'extern' and (not in_pxd)):
                    if not env.is_c_class_scope and (not isinstance(self.base, CPtrDeclaratorNode)):
                        from .ExprNodes import ConstNode
                        self.exception_value = ConstNode(self.pos, value=return_type.exception_value, type=return_type)
            if self.exception_value:
                if self.exception_check == '+':
                    self.exception_value = self.exception_value.analyse_const_expression(env)
                    exc_val_type = self.exception_value.type
                    if not exc_val_type.is_error and (not exc_val_type.is_pyobject) and (not (exc_val_type.is_cfunction and (not exc_val_type.return_type.is_pyobject) and (not exc_val_type.args))) and (not (exc_val_type == PyrexTypes.c_char_type and self.exception_value.value == '*')):
                        error(self.exception_value.pos, 'Exception value must be a Python exception, or C++ function with no arguments, or *.')
                    exc_val = self.exception_value
                else:
                    self.exception_value = self.exception_value.analyse_types(env).coerce_to(return_type, env).analyse_const_expression(env)
                    exc_val = self.exception_value.get_constant_c_result_code()
                    if exc_val is None:
                        error(self.exception_value.pos, 'Exception value must be constant')
                    if not return_type.assignable_from(self.exception_value.type):
                        error(self.exception_value.pos, 'Exception value incompatible with function return type')
                    if visibility != 'extern' and (return_type.is_int or return_type.is_float) and self.exception_value.has_constant_result():
                        try:
                            type_default_value = float(return_type.default_value)
                        except ValueError:
                            pass
                        else:
                            if self.exception_value.constant_result == type_default_value:
                                warning(self.pos, 'Ambiguous exception value, same as default return value: %r' % self.exception_value.constant_result)
            exc_check = self.exception_check
        if return_type.is_cfunction:
            error(self.pos, 'Function cannot return a function')
        func_type = PyrexTypes.CFuncType(return_type, func_type_args, self.has_varargs, optional_arg_count=self.optional_arg_count, exception_value=exc_val, exception_check=exc_check, calling_convention=self.base.calling_convention, nogil=self.nogil, with_gil=self.with_gil, is_overridable=self.overridable, is_const_method=self.is_const_method, templates=self.templates)
        if self.optional_arg_count:
            if func_type.is_fused:

                def declare_opt_arg_struct(func_type, fused_cname):
                    self.declare_optional_arg_struct(func_type, env, fused_cname)
                func_type.declare_opt_arg_struct = declare_opt_arg_struct
            else:
                self.declare_optional_arg_struct(func_type, env)
        callspec = env.directives['callspec']
        if callspec:
            current = func_type.calling_convention
            if current and current != callspec:
                error(self.pos, "cannot have both '%s' and '%s' calling conventions" % (current, callspec))
            func_type.calling_convention = callspec
        if func_type.return_type.is_rvalue_reference:
            warning(self.pos, 'Rvalue-reference as function return type not supported', 1)
        for arg in func_type.args:
            if arg.type.is_rvalue_reference and (not arg.is_forwarding_reference()):
                warning(self.pos, 'Rvalue-reference as function argument not supported', 1)
        return self.base.analyse(func_type, env, visibility=visibility, in_pxd=in_pxd)

    def declare_optional_arg_struct(self, func_type, env, fused_cname=None):
        """
        Declares the optional argument struct (the struct used to hold the
        values for optional arguments). For fused cdef functions, this is
        deferred as analyse_declarations is called only once (on the fused
        cdef function).
        """
        scope = StructOrUnionScope()
        arg_count_member = '%sn' % Naming.pyrex_prefix
        scope.declare_var(arg_count_member, PyrexTypes.c_int_type, self.pos)
        for arg in func_type.args[len(func_type.args) - self.optional_arg_count:]:
            scope.declare_var(arg.name, arg.type, arg.pos, allow_pyobject=True, allow_memoryview=True)
        struct_cname = env.mangle(Naming.opt_arg_prefix, self.base.name)
        if fused_cname is not None:
            struct_cname = PyrexTypes.get_fused_cname(fused_cname, struct_cname)
        op_args_struct = env.global_scope().declare_struct_or_union(name=struct_cname, kind='struct', scope=scope, typedef_flag=0, pos=self.pos, cname=struct_cname)
        op_args_struct.defined_in_pxd = 1
        op_args_struct.used = 1
        func_type.op_arg_struct = PyrexTypes.c_ptr_type(op_args_struct.type)