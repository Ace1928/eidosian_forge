from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_arm_attributes(self):
    self.Elf_Arm_Attribute_Tag = Struct('Elf_Arm_Attribute_Tag', Enum(self.Elf_uleb128('tag'), **ENUM_ATTR_TAG_ARM))