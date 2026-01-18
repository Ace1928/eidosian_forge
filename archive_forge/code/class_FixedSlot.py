from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class FixedSlot(SlotDescriptor):

    def __init__(self, slot_name, value, py3=True, py2=True, ifdef=None):
        SlotDescriptor.__init__(self, slot_name, py3=py3, py2=py2, ifdef=ifdef)
        self.value = value

    def slot_code(self, scope):
        return self.value