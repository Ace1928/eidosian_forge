from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_ntbs(self):
    self.Elf_ntbs = CString