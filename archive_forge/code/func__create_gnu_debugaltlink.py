from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_gnu_debugaltlink(self):
    self.Elf_debugaltlink = Struct('Elf_debugaltlink', CString('sup_filename'), String('sup_checksum', length=20))