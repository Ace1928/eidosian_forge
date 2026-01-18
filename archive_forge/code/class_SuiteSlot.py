from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class SuiteSlot(SlotDescriptor):

    def __init__(self, sub_slots, slot_type, slot_name, substructures, ifdef=None):
        SlotDescriptor.__init__(self, slot_name, ifdef=ifdef)
        self.sub_slots = sub_slots
        self.slot_type = slot_type
        substructures.append(self)

    def is_empty(self, scope):
        for slot in self.sub_slots:
            if slot.slot_code(scope) != '0':
                return False
        return True

    def substructure_cname(self, scope):
        return '%s%s_%s' % (Naming.pyrex_prefix, self.slot_name, scope.class_name)

    def slot_code(self, scope):
        if not self.is_empty(scope):
            return '&%s' % self.substructure_cname(scope)
        return '0'

    def generate_substructure(self, scope, code):
        if not self.is_empty(scope):
            code.putln('')
            if self.ifdef:
                code.putln('#if %s' % self.ifdef)
            code.putln('static %s %s = {' % (self.slot_type, self.substructure_cname(scope)))
            for slot in self.sub_slots:
                slot.generate(scope, code)
            code.putln('};')
            if self.ifdef:
                code.putln('#endif')

    def generate_spec(self, scope, code):
        for slot in self.sub_slots:
            slot.generate_spec(scope, code)