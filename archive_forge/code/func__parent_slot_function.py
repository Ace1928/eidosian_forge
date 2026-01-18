from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def _parent_slot_function(self, scope):
    parent_type_scope = scope.parent_type.base_type.scope
    if scope.parent_scope is parent_type_scope.parent_scope:
        entry = scope.parent_scope.lookup_here(scope.parent_type.base_type.name)
        if entry.visibility != 'extern':
            return self.slot_code(parent_type_scope)
    return None