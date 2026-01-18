from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_leb128(self):
    self.Elf_uleb128 = ULEB128