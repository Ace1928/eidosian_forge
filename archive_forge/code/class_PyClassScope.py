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
class PyClassScope(ClassScope):
    is_py_class_scope = 1

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=False, pytyping_modifiers=None):
        name = self.mangle_class_private_name(name)
        if type is unspecified_type:
            type = py_object_type
        entry = Scope.declare_var(self, name, type, pos, cname=cname, visibility=visibility, api=api, in_pxd=in_pxd, is_cdef=is_cdef, pytyping_modifiers=pytyping_modifiers)
        entry.is_pyglobal = 1
        entry.is_pyclass_attr = 1
        return entry

    def declare_nonlocal(self, name, pos):
        orig_entry = self.lookup_here(name)
        if orig_entry and orig_entry.scope is self and (not orig_entry.from_closure):
            error(pos, "'%s' redeclared as nonlocal" % name)
            orig_entry.already_declared_here()
        else:
            entry = self.lookup(name)
            if entry is None:
                error(pos, "no binding for nonlocal '%s' found" % name)
            else:
                self.entries[name] = entry

    def declare_global(self, name, pos):
        if self.lookup_here(name):
            warning(pos, "'%s' redeclared  ", 0)
        else:
            entry = self.global_scope().lookup_target(name)
            self.entries[name] = entry

    def add_default_value(self, type):
        return self.outer_scope.add_default_value(type)