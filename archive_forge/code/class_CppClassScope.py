from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
class CppClassScope(Scope):
    is_cpp_class_scope = 1
    default_constructor = None
    type = None

    def __init__(self, name, outer_scope, templates=None):
        Scope.__init__(self, name, outer_scope, None)
        self.directives = outer_scope.directives
        self.inherited_var_entries = []
        if templates is not None:
            for T in templates:
                template_entry = self.declare(T, T, PyrexTypes.TemplatePlaceholderType(T), None, 'extern')
                template_entry.is_type = 1

    def declare_var(self, name, type, pos, cname=None, visibility='extern', api=False, in_pxd=False, is_cdef=False, defining=False, pytyping_modifiers=None):
        if not cname:
            cname = name
        self._reject_pytyping_modifiers(pos, pytyping_modifiers)
        entry = self.lookup_here(name)
        if defining and entry is not None:
            if entry.type.same_as(type):
                entry.type = entry.type.with_with_gil(type.with_gil)
            elif type.is_cfunction and type.compatible_signature_with(entry.type):
                entry.type = type
            else:
                error(pos, 'Function signature does not match previous declaration')
        else:
            entry = self.declare(name, cname, type, pos, visibility)
        entry.is_variable = 1
        if type.is_cfunction and self.type:
            if not self.type.get_fused_types():
                entry.func_cname = '%s::%s' % (self.type.empty_declaration_code(), cname)
        if name != 'this' and (defining or name != '<init>'):
            self.var_entries.append(entry)
        return entry

    def declare_cfunction(self, name, type, pos, cname=None, visibility='extern', api=0, in_pxd=0, defining=0, modifiers=(), utility_code=None, overridable=False):
        class_name = self.name.split('::')[-1]
        if name in (class_name, '__init__') and cname is None:
            cname = '%s__init__%s' % (Naming.func_prefix, class_name)
            name = EncodedString('<init>')
            type.return_type = PyrexTypes.CVoidType()
            type.original_args = type.args

            def maybe_ref(arg):
                if arg.type.is_cpp_class and (not arg.type.is_reference):
                    return PyrexTypes.CFuncTypeArg(arg.name, PyrexTypes.c_ref_type(arg.type), arg.pos)
                else:
                    return arg
            type.args = [maybe_ref(arg) for arg in type.args]
        elif name == '__dealloc__' and cname is None:
            cname = '%s__dealloc__%s' % (Naming.func_prefix, class_name)
            name = EncodedString('<del>')
            type.return_type = PyrexTypes.CVoidType()
        if name in ('<init>', '<del>') and type.nogil:
            for base in self.type.base_classes:
                base_entry = base.scope.lookup(name)
                if base_entry and (not base_entry.type.nogil):
                    error(pos, 'Constructor cannot be called without GIL unless all base constructors can also be called without GIL')
                    error(base_entry.pos, 'Base constructor defined here.')
        prev_entry = self.lookup_here(name)
        entry = self.declare_var(name, type, pos, defining=defining, cname=cname, visibility=visibility)
        if prev_entry and (not defining):
            entry.overloaded_alternatives = prev_entry.all_alternatives()
        entry.utility_code = utility_code
        type.entry = entry
        return entry

    def declare_inherited_cpp_attributes(self, base_class):
        base_scope = base_class.scope
        template_type = base_class
        while getattr(template_type, 'template_type', None):
            template_type = template_type.template_type
        if getattr(template_type, 'templates', None):
            base_templates = [T.name for T in template_type.templates]
        else:
            base_templates = ()
        for base_entry in base_scope.inherited_var_entries + base_scope.var_entries:
            if base_entry.name in ('<init>', '<del>'):
                continue
            if base_entry.name in self.entries:
                base_entry.name
            entry = self.declare(base_entry.name, base_entry.cname, base_entry.type, None, 'extern')
            entry.is_variable = 1
            entry.is_inherited = 1
            self.inherited_var_entries.append(entry)
        for base_entry in base_scope.cfunc_entries:
            entry = self.declare_cfunction(base_entry.name, base_entry.type, base_entry.pos, base_entry.cname, base_entry.visibility, api=0, modifiers=base_entry.func_modifiers, utility_code=base_entry.utility_code)
            entry.is_inherited = 1
        for base_entry in base_scope.type_entries:
            if base_entry.name not in base_templates:
                entry = self.declare_type(base_entry.name, base_entry.type, base_entry.pos, base_entry.cname, base_entry.visibility, defining=False)
                entry.is_inherited = 1

    def specialize(self, values, type_entry):
        scope = CppClassScope(self.name, self.outer_scope)
        scope.type = type_entry
        for entry in self.entries.values():
            if entry.is_type:
                scope.declare_type(entry.name, entry.type.specialize(values), entry.pos, entry.cname, template=1)
            elif entry.type.is_cfunction:
                for e in entry.all_alternatives():
                    scope.declare_cfunction(e.name, e.type.specialize(values), e.pos, e.cname, utility_code=e.utility_code)
            else:
                scope.declare_var(entry.name, entry.type.specialize(values), entry.pos, entry.cname, entry.visibility)
        return scope