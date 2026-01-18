from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_gnu_verneed(self):
    self.Elf_Verneed = Struct('Elf_Verneed', self.Elf_half('vn_version'), self.Elf_half('vn_cnt'), self.Elf_word('vn_file'), self.Elf_word('vn_aux'), self.Elf_word('vn_next'))
    self.Elf_Vernaux = Struct('Elf_Vernaux', self.Elf_word('vna_hash'), self.Elf_half('vna_flags'), self.Elf_half('vna_other'), self.Elf_word('vna_name'), self.Elf_word('vna_next'))