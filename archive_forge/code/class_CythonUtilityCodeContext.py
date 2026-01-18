from __future__ import absolute_import
from .TreeFragment import parse_from_strings, StringParseContext
from . import Symtab
from . import Naming
from . import Code
class CythonUtilityCodeContext(StringParseContext):
    scope = None

    def find_module(self, module_name, from_module=None, pos=None, need_pxd=True, absolute_fallback=True, relative_import=False):
        if from_module:
            raise AssertionError('Relative imports not supported in utility code.')
        if module_name != self.module_name:
            if module_name not in self.modules:
                raise AssertionError('Only the cython cimport is supported.')
            else:
                return self.modules[module_name]
        if self.scope is None:
            self.scope = NonManglingModuleScope(self.prefix, module_name, parent_module=None, context=self, cpp=self.cpp)
        return self.scope