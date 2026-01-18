from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_gnu_abi(self):
    self.Elf_abi = Struct('Elf_abi', Enum(self.Elf_word('abi_os'), **ENUM_NOTE_ABI_TAG_OS), self.Elf_word('abi_major'), self.Elf_word('abi_minor'), self.Elf_word('abi_tiny'))