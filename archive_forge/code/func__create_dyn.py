from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_dyn(self):
    d_tag_dict = dict(ENUM_D_TAG_COMMON)
    if self.e_machine in ENUMMAP_EXTRA_D_TAG_MACHINE:
        d_tag_dict.update(ENUMMAP_EXTRA_D_TAG_MACHINE[self.e_machine])
    elif self.e_ident_osabi == 'ELFOSABI_SOLARIS':
        d_tag_dict.update(ENUM_D_TAG_SOLARIS)
    self.Elf_Dyn = Struct('Elf_Dyn', Enum(self.Elf_sxword('d_tag'), **d_tag_dict), self.Elf_xword('d_val'), Value('d_ptr', lambda ctx: ctx['d_val']))