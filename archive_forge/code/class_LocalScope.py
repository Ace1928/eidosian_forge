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
class LocalScope(Scope):
    is_local_scope = True
    has_with_gil_block = False
    _in_with_gil_block = False

    def __init__(self, name, outer_scope, parent_scope=None):
        if parent_scope is None:
            parent_scope = outer_scope
        Scope.__init__(self, name, outer_scope, parent_scope)

    def mangle(self, prefix, name):
        return punycodify_name(prefix + name)

    def declare_arg(self, name, type, pos):
        name = self.mangle_class_private_name(name)
        cname = self.mangle(Naming.var_prefix, name)
        entry = self.declare(name, cname, type, pos, 'private')
        entry.is_variable = 1
        if type.is_pyobject:
            entry.init = '0'
        entry.is_arg = 1
        self.arg_entries.append(entry)
        return entry

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=False, pytyping_modifiers=None):
        name = self.mangle_class_private_name(name)
        if visibility in ('public', 'readonly'):
            error(pos, 'Local variable cannot be declared %s' % visibility)
        entry = Scope.declare_var(self, name, type, pos, cname=cname, visibility=visibility, api=api, in_pxd=in_pxd, is_cdef=is_cdef, pytyping_modifiers=pytyping_modifiers)
        if entry.type.declaration_value:
            entry.init = entry.type.declaration_value
        entry.is_local = 1
        entry.in_with_gil_block = self._in_with_gil_block
        self.var_entries.append(entry)
        return entry

    def declare_global(self, name, pos):
        if self.lookup_here(name):
            warning(pos, "'%s' redeclared  ", 0)
        else:
            entry = self.global_scope().lookup_target(name)
            self.entries[name] = entry

    def declare_nonlocal(self, name, pos):
        orig_entry = self.lookup_here(name)
        if orig_entry and orig_entry.scope is self and (not orig_entry.from_closure):
            error(pos, "'%s' redeclared as nonlocal" % name)
            orig_entry.already_declared_here()
        else:
            entry = self.lookup(name)
            if entry is None or not entry.from_closure:
                error(pos, "no binding for nonlocal '%s' found" % name)

    def _create_inner_entry_for_closure(self, name, entry):
        entry.in_closure = True
        inner_entry = InnerEntry(entry, self)
        inner_entry.is_variable = True
        self.entries[name] = inner_entry
        return inner_entry

    def lookup(self, name):
        entry = Scope.lookup(self, name)
        if entry is not None:
            entry_scope = entry.scope
            while entry_scope.is_comprehension_scope:
                entry_scope = entry_scope.outer_scope
            if entry_scope is not self and entry_scope.is_closure_scope:
                if hasattr(entry.scope, 'scope_class'):
                    raise InternalError('lookup() after scope class created.')
                return self._create_inner_entry_for_closure(name, entry)
        return entry

    def mangle_closure_cnames(self, outer_scope_cname):
        for scope in self.iter_local_scopes():
            for entry in scope.entries.values():
                if entry.from_closure:
                    cname = entry.outer_entry.cname
                    if self.is_passthrough:
                        entry.cname = cname
                    else:
                        if cname.startswith(Naming.cur_scope_cname):
                            cname = cname[len(Naming.cur_scope_cname) + 2:]
                        entry.cname = '%s->%s' % (outer_scope_cname, cname)
                elif entry.in_closure:
                    entry.original_cname = entry.cname
                    entry.cname = '%s->%s' % (Naming.cur_scope_cname, entry.cname)
                    if entry.type.is_cpp_class and entry.scope.directives['cpp_locals']:
                        entry.make_cpp_optional()