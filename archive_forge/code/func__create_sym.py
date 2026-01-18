from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_sym(self):
    st_info_struct = BitStruct('st_info', Enum(BitField('bind', 4), **ENUM_ST_INFO_BIND), Enum(BitField('type', 4), **ENUM_ST_INFO_TYPE))
    st_other_struct = BitStruct('st_other', Enum(BitField('local', 3), **ENUM_ST_LOCAL), Padding(2), Enum(BitField('visibility', 3), **ENUM_ST_VISIBILITY))
    if self.elfclass == 32:
        self.Elf_Sym = Struct('Elf_Sym', self.Elf_word('st_name'), self.Elf_addr('st_value'), self.Elf_word('st_size'), st_info_struct, st_other_struct, Enum(self.Elf_half('st_shndx'), **ENUM_ST_SHNDX))
    else:
        self.Elf_Sym = Struct('Elf_Sym', self.Elf_word('st_name'), st_info_struct, st_other_struct, Enum(self.Elf_half('st_shndx'), **ENUM_ST_SHNDX), self.Elf_addr('st_value'), self.Elf_xword('st_size'))