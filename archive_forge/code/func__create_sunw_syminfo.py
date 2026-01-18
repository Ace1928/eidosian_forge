from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_sunw_syminfo(self):
    self.Elf_Sunw_Syminfo = Struct('Elf_Sunw_Syminfo', Enum(self.Elf_half('si_boundto'), **ENUM_SUNW_SYMINFO_BOUNDTO), self.Elf_half('si_flags'))