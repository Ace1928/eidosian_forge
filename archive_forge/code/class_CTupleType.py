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
class CTupleType(CType):
    is_ctuple = True

    def __init__(self, cname, components):
        self.cname = cname
        self.components = components
        self.size = len(components)
        self.to_py_function = '%s_to_py_%s' % (Naming.convert_func_prefix, self.cname)
        self.from_py_function = '%s_from_py_%s' % (Naming.convert_func_prefix, self.cname)
        self.exception_check = True
        self._convert_to_py_code = None
        self._convert_from_py_code = None
        from .Builtin import tuple_type
        self.equivalent_type = tuple_type

    def __str__(self):
        return '(%s)' % ', '.join((str(c) for c in self.components))

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            return '%s %s' % (str(self), entity_code)
        else:
            return self.base_declaration_code(self.cname, entity_code)

    def can_coerce_to_pyobject(self, env):
        for component in self.components:
            if not component.can_coerce_to_pyobject(env):
                return False
        return True

    def can_coerce_from_pyobject(self, env):
        for component in self.components:
            if not component.can_coerce_from_pyobject(env):
                return False
        return True

    def create_to_py_utility_code(self, env):
        if self._convert_to_py_code is False:
            return None
        if self._convert_to_py_code is None:
            for component in self.components:
                if not component.create_to_py_utility_code(env):
                    self.to_py_function = None
                    self._convert_to_py_code = False
                    return False
            context = dict(struct_type_decl=self.empty_declaration_code(), components=self.components, funcname=self.to_py_function, size=len(self.components))
            self._convert_to_py_code = TempitaUtilityCode.load('ToPyCTupleUtility', 'TypeConversion.c', context=context)
        env.use_utility_code(self._convert_to_py_code)
        return True

    def create_from_py_utility_code(self, env):
        if self._convert_from_py_code is False:
            return None
        if self._convert_from_py_code is None:
            for component in self.components:
                if not component.create_from_py_utility_code(env):
                    self.from_py_function = None
                    self._convert_from_py_code = False
                    return False
            context = dict(struct_type_decl=self.empty_declaration_code(), components=self.components, funcname=self.from_py_function, size=len(self.components))
            self._convert_from_py_code = TempitaUtilityCode.load('FromPyCTupleUtility', 'TypeConversion.c', context=context)
        env.use_utility_code(self._convert_from_py_code)
        return True

    def cast_code(self, expr_code):
        return expr_code