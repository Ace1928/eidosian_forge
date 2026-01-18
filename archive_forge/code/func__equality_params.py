from __future__ import absolute_import
from .TreeFragment import parse_from_strings, StringParseContext
from . import Symtab
from . import Naming
from . import Code
def _equality_params(self):
    outer_scope = self.outer_module_scope
    while isinstance(outer_scope, NonManglingModuleScope):
        outer_scope = outer_scope.outer_scope
    return (self.impl, outer_scope, self.compiler_directives)