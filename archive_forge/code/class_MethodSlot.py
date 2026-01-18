from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class MethodSlot(SlotDescriptor):

    def __init__(self, signature, slot_name, method_name, method_name_to_slot, fallback=None, py3=True, py2=True, ifdef=None, inherited=True):
        SlotDescriptor.__init__(self, slot_name, py3=py3, py2=py2, ifdef=ifdef, inherited=inherited)
        self.signature = signature
        self.slot_name = slot_name
        self.method_name = method_name
        self.alternatives = []
        method_name_to_slot[method_name] = self
        if fallback:
            self.alternatives.append(fallback)
        for alt in (self.py2, self.py3):
            if isinstance(alt, (tuple, list)):
                slot_name, method_name = alt
                self.alternatives.append(method_name)
                method_name_to_slot[method_name] = self

    def slot_code(self, scope):
        entry = scope.lookup_here(self.method_name)
        if entry and entry.is_special and entry.func_cname:
            return entry.func_cname
        for method_name in self.alternatives:
            entry = scope.lookup_here(method_name)
            if entry and entry.is_special and entry.func_cname:
                return entry.func_cname
        return '0'