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
class StructOrUnionScope(Scope):

    def __init__(self, name='?'):
        Scope.__init__(self, name, outer_scope=None, parent_scope=None)

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=False, pytyping_modifiers=None, allow_pyobject=False, allow_memoryview=False, allow_refcounted=False):
        if not cname:
            cname = name
            if visibility == 'private':
                cname = c_safe_identifier(cname)
        if type.is_cfunction:
            type = PyrexTypes.CPtrType(type)
        self._reject_pytyping_modifiers(pos, pytyping_modifiers)
        entry = self.declare(name, cname, type, pos, visibility)
        entry.is_variable = 1
        self.var_entries.append(entry)
        if type.is_pyobject:
            if not allow_pyobject:
                error(pos, 'C struct/union member cannot be a Python object')
        elif type.is_memoryviewslice:
            if not allow_memoryview:
                error(pos, 'C struct/union member cannot be a memory view')
        elif type.needs_refcounting:
            if not allow_refcounted:
                error(pos, "C struct/union member cannot be reference-counted type '%s'" % type)
        return entry

    def declare_cfunction(self, name, type, pos, cname=None, visibility='private', api=0, in_pxd=0, defining=0, modifiers=(), overridable=False):
        if overridable:
            error(pos, "C struct/union member cannot be declared 'cpdef'")
        return self.declare_var(name, type, pos, cname=cname, visibility=visibility)