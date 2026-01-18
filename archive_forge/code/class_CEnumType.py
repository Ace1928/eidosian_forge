from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
class CEnumType(CIntLike, CType, EnumMixin):
    is_enum = 1
    signed = 1
    rank = -1

    def __init__(self, name, cname, typedef_flag, namespace=None, doc=None):
        self.name = name
        self.doc = doc
        self.cname = cname
        self.values = []
        self.typedef_flag = typedef_flag
        self.namespace = namespace
        self.default_value = '(%s) 0' % self.empty_declaration_code()

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<CEnumType %s %s%s>' % (self.name, self.cname, ('', ' typedef')[self.typedef_flag])

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            base_code = self.name
        else:
            if self.namespace:
                base_code = '%s::%s' % (self.namespace.empty_declaration_code(), self.cname)
            elif self.typedef_flag:
                base_code = self.cname
            else:
                base_code = 'enum %s' % self.cname
            base_code = public_decl(base_code, dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def specialize(self, values):
        if self.namespace:
            namespace = self.namespace.specialize(values)
            if namespace != self.namespace:
                return CEnumType(self.name, self.cname, self.typedef_flag, namespace)
        return self

    def create_type_wrapper(self, env):
        from .UtilityCode import CythonUtilityCode
        old_to_py_function = self.to_py_function
        self.to_py_function = None
        CIntLike.create_to_py_utility_code(self, env)
        enum_to_pyint_func = self.to_py_function
        self.to_py_function = old_to_py_function
        env.use_utility_code(CythonUtilityCode.load('EnumType', 'CpdefEnums.pyx', context={'name': self.name, 'items': tuple(self.values), 'enum_doc': self.doc, 'enum_to_pyint_func': enum_to_pyint_func, 'static_modname': env.qualified_name}, outer_module_scope=env.global_scope()))

    def create_to_py_utility_code(self, env):
        if self.to_py_function is not None:
            return self.to_py_function
        if not self.entry.create_wrapper:
            return super(CEnumType, self).create_to_py_utility_code(env)
        self.create_enum_to_py_utility_code(env)
        return True