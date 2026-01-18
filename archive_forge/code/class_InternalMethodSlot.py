from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class InternalMethodSlot(SlotDescriptor):

    def __init__(self, slot_name, **kargs):
        SlotDescriptor.__init__(self, slot_name, **kargs)

    def slot_code(self, scope):
        return scope.mangle_internal(self.slot_name)