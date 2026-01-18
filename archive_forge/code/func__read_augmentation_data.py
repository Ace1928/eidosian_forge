import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
def _read_augmentation_data(self, entry_structs):
    """ Read augmentation data.

        This assumes that the augmentation string starts with 'z', i.e. that
        augmentation data is prefixed by a length field, which is not returned.
        """
    if not self.for_eh_frame:
        return b''
    augmentation_data_length = struct_parse(Struct('Dummy_Augmentation_Data', entry_structs.Dwarf_uleb128('length')), self.stream)['length']
    return self.stream.read(augmentation_data_length)