from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_phdr(self):
    p_type_dict = ENUM_P_TYPE_BASE
    if self.e_machine == 'EM_ARM':
        p_type_dict = ENUM_P_TYPE_ARM
    elif self.e_machine == 'EM_AARCH64':
        p_type_dict = ENUM_P_TYPE_AARCH64
    elif self.e_machine == 'EM_MIPS':
        p_type_dict = ENUM_P_TYPE_MIPS
    elif self.e_machine == 'EM_RISCV':
        p_type_dict = ENUM_P_TYPE_RISCV
    if self.elfclass == 32:
        self.Elf_Phdr = Struct('Elf_Phdr', Enum(self.Elf_word('p_type'), **p_type_dict), self.Elf_offset('p_offset'), self.Elf_addr('p_vaddr'), self.Elf_addr('p_paddr'), self.Elf_word('p_filesz'), self.Elf_word('p_memsz'), self.Elf_word('p_flags'), self.Elf_word('p_align'))
    else:
        self.Elf_Phdr = Struct('Elf_Phdr', Enum(self.Elf_word('p_type'), **p_type_dict), self.Elf_word('p_flags'), self.Elf_offset('p_offset'), self.Elf_addr('p_vaddr'), self.Elf_addr('p_paddr'), self.Elf_xword('p_filesz'), self.Elf_xword('p_memsz'), self.Elf_xword('p_align'))