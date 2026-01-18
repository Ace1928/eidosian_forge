from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class GCDependentSlot(InternalMethodSlot):

    def __init__(self, slot_name, **kargs):
        InternalMethodSlot.__init__(self, slot_name, **kargs)

    def slot_code(self, scope):
        if not scope.needs_gc():
            return '0'
        if not scope.has_cyclic_pyobject_attrs:
            parent_type_scope = scope.parent_type.base_type.scope
            if scope.parent_scope is parent_type_scope.parent_scope:
                entry = scope.parent_scope.lookup_here(scope.parent_type.base_type.name)
                if entry.visibility != 'extern':
                    return self.slot_code(parent_type_scope)
        return InternalMethodSlot.slot_code(self, scope)