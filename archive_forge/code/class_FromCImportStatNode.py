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
class FromCImportStatNode(StatNode):
    child_attrs = []
    module_name = None
    relative_level = None
    imported_names = None

    def analyse_declarations(self, env):
        if not env.is_module_scope:
            error(self.pos, 'cimport only allowed at module level')
            return
        qualified_name_components = env.qualified_name.count('.') + 1
        if self.relative_level:
            if self.relative_level > qualified_name_components:
                error(self.pos, 'relative cimport beyond main package is not allowed')
                return
            elif self.relative_level == qualified_name_components and (not env.is_package):
                error(self.pos, 'relative cimport from non-package directory is not allowed')
                return
        module_scope = env.find_module(self.module_name, self.pos, relative_level=self.relative_level)
        if not module_scope:
            return
        module_name = module_scope.qualified_name
        env.add_imported_module(module_scope)
        for pos, name, as_name in self.imported_names:
            if name == '*':
                for local_name, entry in list(module_scope.entries.items()):
                    env.add_imported_entry(local_name, entry, pos)
            else:
                entry = module_scope.lookup(name)
                if entry:
                    entry.used = 1
                else:
                    is_relative_import = self.relative_level is not None and self.relative_level > 0
                    submodule_scope = env.context.find_module(name, from_module=module_scope, pos=self.pos, absolute_fallback=False, relative_import=is_relative_import)
                    if not submodule_scope:
                        continue
                    if submodule_scope.parent_module is module_scope:
                        env.declare_module(as_name or name, submodule_scope, self.pos)
                    else:
                        error(pos, "Name '%s' not declared in module '%s'" % (name, module_name))
                if entry:
                    local_name = as_name or name
                    env.add_imported_entry(local_name, entry, pos)
        if module_name.startswith('cpython') or module_name.startswith('cython'):
            if module_name in utility_code_for_cimports:
                env.use_utility_code(utility_code_for_cimports[module_name]())
            for _, name, _ in self.imported_names:
                fqname = '%s.%s' % (module_name, name)
                if fqname in utility_code_for_cimports:
                    env.use_utility_code(utility_code_for_cimports[fqname]())

    def declaration_matches(self, entry, kind):
        if not entry.is_type:
            return 0
        type = entry.type
        if kind == 'class':
            if not type.is_extension_type:
                return 0
        else:
            if not type.is_struct_or_union:
                return 0
            if kind != type.kind:
                return 0
        return 1

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        if self.module_name == 'numpy':
            cimport_numpy_check(self, code)