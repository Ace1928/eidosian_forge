import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
def _parse_lsda_pointer(self, structs, stream_offset, encoding):
    """ Parse bytes to get an LSDA pointer.

        The basic encoding (lower four bits of the encoding) describes how the values are encoded in a CIE or an FDE.
        The modifier (upper four bits of the encoding) describes how the raw values, after decoded using a basic
        encoding, should be modified before using.

        Ref: https://www.airs.com/blog/archives/460
        """
    assert encoding != DW_EH_encoding_flags['DW_EH_PE_omit']
    basic_encoding = encoding & 15
    modifier = encoding & 240
    formats = self._eh_encoding_to_field(structs)
    ptr = struct_parse(Struct('Augmentation_Data', formats[basic_encoding]('LSDA_pointer')), self.stream, stream_pos=stream_offset)['LSDA_pointer']
    if modifier == DW_EH_encoding_flags['DW_EH_PE_absptr']:
        pass
    elif modifier == DW_EH_encoding_flags['DW_EH_PE_pcrel']:
        ptr += self.address + stream_offset
    else:
        assert False, 'Unsupported encoding modifier for LSDA pointer: {:#x}'.format(modifier)
    return ptr