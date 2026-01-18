import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
def _parse_cie_augmentation(self, header, entry_structs):
    """ Parse CIE augmentation data from the annotation string in `header`.

        Return a tuple that contains 1) the augmentation data as a string
        (without the length field) and 2) the augmentation data as a dict.
        """
    augmentation = header.get('augmentation')
    if not augmentation:
        return ('', {})
    assert augmentation.startswith(b'z'), 'Unhandled augmentation string: {}'.format(repr(augmentation))
    available_fields = {b'z': entry_structs.Dwarf_uleb128('length'), b'L': entry_structs.Dwarf_uint8('LSDA_encoding'), b'R': entry_structs.Dwarf_uint8('FDE_encoding'), b'S': True, b'P': Struct('personality', entry_structs.Dwarf_uint8('encoding'), Switch('function', lambda ctx: ctx.encoding & 15, {enc: fld_cons('function') for enc, fld_cons in self._eh_encoding_to_field(entry_structs).items()}))}
    fields = []
    aug_dict = {}
    for b in iterbytes(augmentation):
        try:
            fld = available_fields[b]
        except KeyError:
            break
        if fld is True:
            aug_dict[fld] = True
        else:
            fields.append(fld)
    offset = self.stream.tell()
    struct = Struct('Augmentation_Data', *fields)
    aug_dict.update(struct_parse(struct, self.stream, offset))
    self.stream.seek(offset)
    aug_bytes = self._read_augmentation_data(entry_structs)
    return (aug_bytes, aug_dict)