import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
class FDE(CFIEntry):

    def __init__(self, header, structs, instructions, offset, augmentation_bytes=None, cie=None, lsda_pointer=None):
        super(FDE, self).__init__(header, structs, instructions, offset, augmentation_bytes=augmentation_bytes, cie=cie)
        self.lsda_pointer = lsda_pointer